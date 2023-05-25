# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb
from training.sampler import edm_sampler, StackedRandomGenerator
from training.structure import Structure

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    embedder_kwargs     = {},       # Options for embedder.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    structure_kwargs    = {},       # Options for structure (which variables exist, which are observed).
    emb_structure_kwargs    = {},       # Options for structure (which variables exist, which are observed).
    structured_kwargs   = {},       # Options for structured (i.e. per-variable) arguments.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    equally_weight_tensors = False, # Equally weight mean loss over each tensor when training joint models
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 50,      # How often to dump training state, None = disable.
    log_img_ticks       = 50,       # How often to log images, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_state_dict   = None,     # Specify state_dict to load from instead of pickle
    dump_state_dict   = None,     # Just dump state_dict to file and exit
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    just_visualize      = False,    # Just visualize the dataset and exit.
    freeze_pretrained   = False,    # Freeze pretrained layers (assumes pretrained layers exist if set)
    freeze_embedder     = False,    # Freeze embedder
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct objects describing problem structure.
    structure = Structure(**structure_kwargs, dataset=dataset_obj)

    # Construct embedder
    embedder = dnnlib.util.construct_class_by_name(**embedder_kwargs)
    embedded_structure = structure.get_embedded_version(
        embed_func=embedder, new_exist=emb_structure_kwargs["emb_exist"],
        new_observed=emb_structure_kwargs["emb_observed"],
    )

    # Set up loss function.
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs, structure=embedded_structure)  # training.loss.(VP|VE|EDM)Loss
    jump_structure = loss_fn.structure

    # Construct network.
    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs, structure=jump_structure, embedder=embedder) # subclass of torch.nn.Module
    net.preaugment_structure = structure
    """
    We now have three structures:
    - structure: the structure of the original problem
    - embedded_structure: the structure of the problem after adding embedding dims
     (same as structure with the default --embedder NullEmbedder option)
    - jump_structure: the structure of the problem augmented with dimension prediction variables
    """
    net.train().to(device)
    net.model.set_requires_grad(True, freeze_pretrained=freeze_pretrained)
    if freeze_embedder:
        net.embedder.set_requires_grad(False)
    if dist.get_rank() == 0:
        wandb.log({'num_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad)})

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    if hasattr(dataset_obj, 'embedder') and dataset_obj.save_embedder:
        params = list(net.parameters()) + list(dataset_obj.embedder.parameters())
    else:
        params = net.parameters()
    optimizer = dnnlib.util.construct_class_by_name(params=params, **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval()
    ema.requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes firstw
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory
    elif resume_state_dict:
        state_dict = torch.load(resume_state_dict, map_location=torch.device('cpu'))
        ema.load_state_dict(state_dict)
        net.load_state_dict(state_dict)
    if dump_state_dict:
        torch.save(ema.state_dict(), os.path.join(run_dir, 'state_dict.pt'))
        exit()

    if just_visualize:
        cluster_dir = os.path.join(run_dir, 'clusters')
        os.makedirs(cluster_dir, exist_ok=True)
        dataset_obj.vis_clusters(dirname=cluster_dir, augment_pipe=augment_pipe, device=device)
        exit()

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # add paths to tell network/loss function to make plots
        if dist.get_rank() == 0:
            plot_dir = os.path.join(run_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            net.model.plot_path = os.path.join(plot_dir, f'training-{cur_tick//10:06d}-inputs.png')
            loss_fn.plot_path = os.path.join(plot_dir, f'training-{cur_tick//10:06d}-outputs.png')

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data = next(dataset_iterator)
                if hasattr(dataset_obj, 'duplicate_videos_in_batch'):
                    data = tuple(t.flatten(end_dim=1) for t in data)  # flattent duplicates all into batch dim
                data = tuple(t.to(device) for t in data)
                data, augment_labels = dataset_obj.augment(data, augment_pipe)
                data = embedder(data)
                x, y = embedded_structure.flatten_batch(data, contains_marg=True)
                loss = loss_fn(net=ddp, x=x, y=y, augment_labels=augment_labels)
                training_stats.report('Loss/loss', loss)
                loss_per_latent_tensor = jump_structure.unflatten_latents(loss)
                for name, tensor_loss in zip(jump_structure.latent_names, loss_per_latent_tensor):
                    training_stats.report(f'Loss/{name}', tensor_loss)
                if equally_weight_tensors:
                    scalar_loss = loss_scaling * sum(tensor_loss.mean() for tensor_loss in loss_per_latent_tensor)
                else:
                    scalar_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                training_stats.report('Loss/scalar_value', scalar_loss)
                scalar_loss.backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            if hasattr(dataset_obj, 'embedder') and dataset_obj.save_embedder:
                data['embedder'] = dataset_obj.get_network_state_dict()
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                torch.save(ema.state_dict(), os.path.join(run_dir, f'state-dict-{cur_nimg//1000:06d}.pt'))
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            wandb_dict = {}
            for field, values in training_stats.default_collector.as_dict().items():
                wandb_dict[field] = values["mean"]
                if values["num"] > 1:
                    wandb_dict[field + "_std"] = values["std"]
            wandb.log(wandb_dict)
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Sample and log images.
        # if (dist.get_rank() == 0) and (log_img_ticks is not None) and (done or cur_tick % log_img_ticks == 0):
        #     with torch.no_grad():
        #         sampling_batch_size = 1
        #         rnd = StackedRandomGenerator(device, seeds=np.arange(sampling_batch_size))
        #         xT = rnd.randn([sampling_batch_size, embedded_structure.latent_dim], device=device)
        #         indices = rnd.randint(len(dataset_obj), size=[sampling_batch_size, 1], device=device).squeeze(1)
        #         unstacked_data = [dataset_obj.__getitem__(i.item(), will_augment=False, deterministic=True, do_duplicate=False) for i in indices]
        #         data = tuple(torch.stack([datum[t] for datum in unstacked_data]).to(device) for t in range(len(unstacked_data[0])))
        #         data = embedder(data)
        #         _, y = embedded_structure.flatten_batch(data, contains_marg=True)
        #         x0, y = edm_sampler(ema, xT, y, randn_like=rnd.randn_like, dim_deletion_process=loss_fn.dim_deletion_process)
        #         tensors = embedded_structure.unflatten_batch(x0, y, pad_marg=True)
        #         tensors, emb_tensors = tensors[:len(structure.shapes)], tensors[len(structure.shapes):]
        #         dataset_obj.log_batch(tensors, emb_tensors)
        
        # log dataset images on first iteration
        if (dist.get_rank() == 0) and (log_img_ticks is not None) and (cur_tick == 0):
            log_img_count = 9
            rnd = StackedRandomGenerator(device, seeds=np.arange(log_img_count))
            indices = rnd.randint(len(dataset_obj), size=[log_img_count, 1], device=device).squeeze(1)
            unstacked_data = [dataset_obj.__getitem__(i.item(), will_augment=False, deterministic=True, do_duplicate=False) for i in indices]
            data = tuple(torch.stack([datum[t] for datum in unstacked_data]).to(device) for t in range(len(unstacked_data[0])))
            data = embedder(data)
            tensors, emb_tensors = data[:len(structure.shapes)], data[len(structure.shapes):]
            dataset_obj.log_batch(tensors, emb_tensors)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
