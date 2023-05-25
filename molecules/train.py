# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training.dataset import datasets_to_kwargs, kwargs_gettable_from_dataset
from training.loss import losses_to_kwargs
from training.sampler import samplers_to_kwargs
from training.grad_conditioning import grad_conditioners_to_kwargs
from training.networks import networks_to_kwargs
import wandb
import glob
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    print('parse_int_list', s)
    if isinstance(s, tuple): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return tuple(ranges)

def parse_float_list(s):
    if isinstance(s, tuple): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(float(m.group(1)), float(m.group(2))+1))
        else:
            ranges.append(float(p))
    return tuple(ranges)

def str2bool(s):
    return 't' in s.lower()

#----------------------------------------------------------------------------

added_names = set()
def add_specific_options(f, classes_to_kwargs):
    for key in classes_to_kwargs.keys():
        new_set = set()
        for inner_tuple in classes_to_kwargs[key]:
            new_set.add((key.lower() + '_' + inner_tuple[0], inner_tuple[1], inner_tuple[2]))
        classes_to_kwargs[key] = new_set

    class_specific_kwargs = set.union(*classes_to_kwargs.values())
    for name, type_str, default in class_specific_kwargs:
        assert not any(c.isupper() for c in name) # not allowed capital letters!
        if name in added_names:
            raise ValueError(f'{name} parameter appears twice so will override one!!!')
        else:
            added_names.add(name)
        f = click.option(f'--{name}', help=f'{name} for dataset', type=eval(type_str), default=default, show_default=True)(f)
    return f

def dataset_specific_options(f):
    return add_specific_options(f, datasets_to_kwargs)

def loss_specific_options(f):
    return add_specific_options(f, losses_to_kwargs)

def sampler_specific_options(f):
    return add_specific_options(f, samplers_to_kwargs)

def grad_conditioning_specific_options(f):
    return add_specific_options(f, grad_conditioners_to_kwargs)

def network_specific_options(f):
    return add_specific_options(f, networks_to_kwargs)


@click.command()

@click.option('--exist',          help='List of 1s/0s to specify which tensors to use (i.e. to not marginalise).', metavar='LIST',  type=parse_int_list, default=None, show_default=True)  # TODO implement for any dataset
@click.option('--observed',      help='Which dataset tensors are observed', metavar='LIST',         type=parse_int_list, required=True)
# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, default='training-runs', show_default=True)
@click.option('--data_class',    help='Dataset class to use',                                       type=click.Choice(datasets_to_kwargs.keys()), required=True)
@click.option('--precond',       help='Preconditioning',                            type=click.Choice(['edm', 'eps', 'x0', 'none']), default='eps', show_default=True)
@click.option('--loss_class',     help='Loss class to use',                                          type=click.Choice(losses_to_kwargs.keys()), required=True )
@click.option('--just_visualize', help='Whether to just visualize the dataset.', metavar='BOOL',    type=bool, default=False, show_default=True)
@click.option('--noise_embed',    help='How to embed timestep for the model.', metavar='STR',       type=click.Choice(['ts', 'ts*1000', 'edm']), default='ts*1000', show_default=True)
@click.option('--sampler_class',  help='Sampler class to use',                metavar='STR',        type=click.Choice(samplers_to_kwargs.keys()), required=True)
@click.option('--grad_conditioner_class', help='Gradient conditioning class to use', metavar='STR', type=click.Choice(grad_conditioners_to_kwargs.keys()), default='EDM', show_default=True)
@click.option('--distributed',    help='Whether to use distributed training', metavar='BOOL',       type=bool, default=False, show_default=True)

# @click.option('--noise_mult',    help='Multipliers for amount of noise added to each tensor.', metavar='LIST', type=parse_float_list, default=(1.,))

# Architecture options.
# @click.option('--arch',          help='Network architecture',                                       type=click.Choice(['concatunet', 'gsdm', 'test', 'egnn', 'egnn_cont', 'egnn_jump', 'mol_mlp', 'graph_transformer']), default='concatunet', show_default=True)
@click.option('--network_class',          help='Network architecture',                                       type=click.Choice(networks_to_kwargs.keys()), required=True)

# @click.option('--pred_x0',       help='List describing whether to predict x0 for each tensor.', metavar='LIST', type=parse_int_list, default=(0,))
# @click.option('--rate_use_x0_pred', help = 'Parameterize rate network through x0 dim prediction', metavar='BOOL', type=bool, default=True, show_default=True)

# @click.option('--softmax_onehot', help='Whether to use softmax on model output for onehots. Only makes sense with --pred_x0 on for the onehots.', metavar='BOOL', type=bool, default=False, show_default=True)
# @click.option('--channel_mult_emb', help='Channel multiplier for vector embeddings.', metavar='INT', type=int, default=4, show_default=True)
# @click.option('--channel_mult_noise', help='Channel multiplier for initial noise (and label+vector) embeddings.', metavar='INT', type=int, default=4, show_default=True)
# @click.option('--sparse_attention', help='Whether to use sparse attention', metavar='BOOL', type=bool, default=True, show_default=True)
# @click.option('--model_channels', help='number of channels in transformer', metavar='INT', type=int, default=128)
# @click.option('--num_blocks', help='number of blocks in transformer', metavar='INT', type=int, default=4)

@dataset_specific_options

@loss_specific_options

@sampler_specific_options

@grad_conditioning_specific_options

@network_specific_options

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--sample',        help='How often to log images', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--resume_latest_from_resume_id', help='Resume from the latest training state in the directoy given by resume_id', is_flag=True)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--device', type=str, default='cuda')

# wandb
@click.option('--wandb_dir',     help='Where to save the wandb results', metavar='DIR',             type=str, default='.')
@click.option('--resume_id',     help='Wandb id to resume from', metavar='ID',                      type=str, default=None)
@click.option('--set_wandb_id',  help='Optional way to set the wandb id', metavar='ID',             type=str, default=None)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)

    # torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method('fork')
    if opts.distributed:
        dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    dataset_class_name = 'training.dataset.' + opts.data_class
    c.dataset_kwargs = dnnlib.EasyDict(class_name=dataset_class_name)
    for kwarg_name, _, _ in datasets_to_kwargs[opts.data_class]:
        new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
        c.dataset_kwargs[new_kwarg_name] = opts[kwarg_name]

    c.distributed = opts.distributed
    c.device = opts.device
    if opts.workers == 0:
        print('WARNING using 0 workers which was previously disallowed by EDM')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    loss_class_name = 'training.loss.' + opts.loss_class
    c.loss_kwargs = dnnlib.EasyDict(class_name=loss_class_name)
    for kwarg_name, _, _ in losses_to_kwargs[opts.loss_class]:
        # remove the class name from the start of the kwarg name
        # safe to do now since we have selected only one from the options
        new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
        c.loss_kwargs[new_kwarg_name] = opts[kwarg_name]


    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.just_visualize = opts.just_visualize

    # Validate dataset options.
    try:
        pass
        dataset_name = opts.data_class
        # dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        # dataset_name = dataset_obj.name
        # for kwarg_name, getter in kwargs_gettable_from_dataset[opts.data_class]:
        #     # sets e.g. max_size and resolution for image datasets
        #     setattr(c.dataset_kwargs, kwarg_name, getter(dataset_obj))  # sets
        # del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


    c.structure_kwargs = dnnlib.EasyDict(exist=opts.exist, observed=opts.observed)

    sampler_class_name = 'training.sampler.' + opts.sampler_class
    c.sampler_kwargs = dnnlib.EasyDict(class_name=sampler_class_name)
    for kwarg_name, _, _ in samplers_to_kwargs[opts.sampler_class]:
        new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
        c.sampler_kwargs[new_kwarg_name] = opts[kwarg_name]

    grad_conditioner_class_name = 'training.grad_conditioning.' + opts.grad_conditioner_class
    c.grad_conditioner_kwargs = dnnlib.EasyDict(class_name=grad_conditioner_class_name)
    for kwarg_name, _, _ in grad_conditioners_to_kwargs[opts.grad_conditioner_class]:
        new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
        c.grad_conditioner_kwargs[new_kwarg_name] = opts[kwarg_name]


    # Network architecture.
    c.network_kwargs = dnnlib.EasyDict(model_type=opts.network_class)
    for kwarg_name, _, _ in networks_to_kwargs[opts.network_class]:
        new_kwarg_name = "_".join(kwarg_name.split("_")[1:])
        c.network_kwargs[new_kwarg_name] = opts[kwarg_name]


    # Preconditioning & loss function.
    c.network_kwargs.update(noise_embed=opts.noise_embed, use_fp16=opts.fp16)
    if opts.precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    elif opts.precond == 'eps':
        c.network_kwargs.class_name = 'training.networks.EpsilonPrecond'
    elif opts.precond == 'x0':
        c.network_kwargs.class_name = 'training.networks.X0Precond'
    elif opts.precond == 'none':
        c.network_kwargs.class_name = 'training.networks.NonePrecond'
    else:
        raise NotImplementedError(opts.precond)

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump, log_img_ticks=opts.sample)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=c.device)
        if opts.distributed:
            torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)
    dist.print0('seed: ', c.seed)

    # Transfer learning and resume.
    if opts.resume_latest_from_resume_id:
        assert opts.resume_id is not None
        training_states = glob.glob(Path(opts.outdir).joinpath(str(opts.resume_id)).joinpath('training-state-*.pt').as_posix())
        training_states = sorted(training_states)
        opts.resume = training_states[-1]
        dist.print0(f'Resuming from {opts.resume}')


    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Initialize wandb
    if dist.get_rank() == 0:
        wandb_id = None
        if opts.resume_id is not None:
            wandb_id = opts.resume_id
        if opts.set_wandb_id is not None:
            wandb_id = opts.set_wandb_id
        if opts.resume_id is not None and opts.set_wandb_id is not None:
            assert opts.resume_id == opts.set_wandb_id

        wandb.init(
            entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'], config=c,
            dir=opts.wandb_dir, id=wandb_id, resume=opts.resume_id is not None,
        )

    # Description string.
    cond_str = 'cond-' + ''.join(map(str, opts.observed))
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.network_class:s}-edm-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{wandb.run.id}')
        # assert not os.path.exists(c.run_dir)

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
