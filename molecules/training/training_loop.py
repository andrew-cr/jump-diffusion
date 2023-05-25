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
from training.sampler import StackedRandomGenerator
from training.structure import Structure, StructuredDataBatch
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    sampler_kwargs      = {},       # Options for the sampler
    grad_conditioner_kwargs = {},   # Options for gradient conditioning
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    structure_kwargs    = {},       # Options for structure (which variables exist, which are observed).
    structured_kwargs   = {},       # Options for structured (i.e. per-variable) arguments.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    # lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    log_img_ticks       = 50,       # How often to log images, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    just_visualize      = False,    # Just visualize the dataset and exit.
    distributed         = False,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False

    if device != 'cpu':
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
    train_dataset_kwargs = copy.deepcopy(dataset_kwargs)
    train_dataset_kwargs['train_or_valid'] = 'train'
    dataset_obj = dnnlib.util.construct_class_by_name(**train_dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    valid_dataset_kwargs = copy.deepcopy(dataset_kwargs)
    valid_dataset_kwargs['train_or_valid'] = 'valid'
    valid_dataset_obj = dnnlib.util.construct_class_by_name(**valid_dataset_kwargs)

    # Construct objects describing problem structure.
    structure = Structure(**structure_kwargs, dataset=dataset_obj)

    # Construct network.
    dist.print0('Constructing network...')
    net = dnnlib.util.construct_class_by_name(**network_kwargs, structure=structure) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            # sampled_problem_dim, *data = dataset_obj[0]
            # unstacked_data = [data] * batch_gpu
            # data = tuple(torch.stack(d).to(device) for d in zip(*unstacked_data))
            # structure.set_varying_problem_dims(torch.tensor([sampled_problem_dim]*batch_gpu), sampled_problem_dim)
            # data = structure.graphical_structure.strip_padding(data, sampled_problem_dim)
            # x, y = structure.flatten_batch(data, contains_marg=False)
            # sigma = torch.ones([batch_gpu], device=device)
            # inputs = [x, y, sigma]
            # misc.print_module_summary(net, inputs, max_nesting=1)
            wandb.log({'num_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad)})
            print('num parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Setup sampler
    sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs, structure=structure)

    # Setup gradient conditioning
    grad_conditioner = dnnlib.util.construct_class_by_name(**grad_conditioner_kwargs)


    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs, structure=structure)  # training.loss.(VP|VE|EDM)Loss
    if hasattr(dataset_obj, 'embedder') and dataset_obj.save_embedder:
        params = list(net.parameters()) + list(dataset_obj.embedder.parameters())
    else:
        params = net.parameters()
    optimizer = dnnlib.util.construct_class_by_name(params=params, **optimizer_kwargs) # subclass of torch.optim.Optimizer
    if distributed:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    else:
        ddp = net
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # ensure structure object is shared so resetting dims on one updates the other -----------------
    if hasattr(ema, 'noise_mult'):
        ema.noise_mult.structure = structure
    if hasattr(ema, 'pred_x0'):
        ema.pred_x0.structure = structure
    ema.structure = structure
    ema.model.structure = structure
    # ----------------------------------------------------------------------------------------------

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0 and distributed:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0 and distributed:
            dist.print0('Rank 0 waiting for others to load pickle')
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        if hasattr(dataset_obj, 'embedder'):
            dataset_obj.load_network_state_dict(data['embedder'])
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

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

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                # if cur_nimg > batch_size * 10:
                #     dims, *data = next(dataset_iterator)
                #     st_batch = StructuredDataBatch(data, dims, structure_kwargs.observed,
                #         structure_kwargs.exist, dataset_obj.is_onehot, structure.graphical_structure
                #     )
                #     st_batch.to(device)
                #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #         with record_function("loss_fn"):
                #             loss, loss_dict = loss_fn(net=ddp, st_batch=st_batch)

                #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

                #     assert False


                dims, *data = next(dataset_iterator)
                st_batch = StructuredDataBatch(data, dims, structure_kwargs.observed,
                    structure_kwargs.exist, dataset_obj.is_onehot, structure.graphical_structure
                )
                st_batch.to(device)
                loss, loss_dict = loss_fn(net=ddp, st_batch=st_batch)
                training_stats.report('Loss/loss', loss)
                for name, tensor_loss in loss_dict.items():
                    if tensor_loss is not None:
                        training_stats.report(f'Loss/{name}', tensor_loss)

                # loss_per_latent_tensor = structure.unflatten_latents(loss)
                # for name, tensor_loss in zip(structure.latent_names, loss_per_latent_tensor):
                #     training_stats.report(f'Loss/{name}', tensor_loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        grad_conditioner.condition(optimizer, net, cur_nimg, optimizer_kwargs['lr'])
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        for b_ema, b_net in zip(ema.buffers(), net.buffers()):
            b_ema.copy_(b_net.detach())

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        try:
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"iters/sec {training_stats.report0('Timing/iters_per_sec', ((cur_nimg - tick_start_nimg)/batch_size)/ (tick_end_time - tick_start_time))}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            if device != 'cpu':
                fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
                fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        except Exception as e:
            print('Exception whilst logging stats: ', e)

        if device != 'cpu':
            torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, dataset_kwargs=dict(dataset_kwargs))
            if hasattr(dataset_obj, 'embedder') and dataset_obj.save_embedder:
                data['embedder'] = dataset_obj.get_network_state_dict()
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if distributed:
                        misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
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
        if (dist.get_rank() == 0) and (log_img_ticks is not None) and (done or cur_tick % log_img_ticks == 0):
            with torch.no_grad():
                rnd = StackedRandomGenerator(device, seeds=np.arange(batch_gpu))
                indices = rnd.randint(len(valid_dataset_obj), size=[batch_gpu, 1], device=device).squeeze(1)
                unstacked_data = [valid_dataset_obj.__getitem__(i.item(), will_augment=False) for i in indices]
                unstacked_data_no_dims = [d[1:] for d in unstacked_data]
                dims = torch.tensor([d[0] for d in unstacked_data])
                data = tuple(torch.stack(d).to(device) for d in zip(*unstacked_data_no_dims))
                st_batch = StructuredDataBatch(data, dims, structure_kwargs.observed,
                    structure_kwargs.exist, valid_dataset_obj.is_onehot, structure.graphical_structure
                )
                x0_st_batch = sampler.sample(ema, st_batch, loss_fn, rnd)
                dataset_obj.log_batch(in_st_batch=st_batch, out_st_batch=x0_st_batch)

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
