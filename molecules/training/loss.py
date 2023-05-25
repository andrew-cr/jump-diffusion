# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F
from torch_utils import persistence
from training.structure import StructuredArgument, StructuredDataBatch
from scipy.stats import poisson
import numpy as np
from training.diffusion_utils import VP_SDE, ConstForwardRate, StepForwardRate
import itertools as it
from torch_utils import training_stats
import math
from training.egnn_utils import EnVariationalDiffusion, DistributionNodes, remove_mean_with_mask, \
    check_mask_correct, assert_mean_zero_with_mask, compute_loss_and_nll, sample_gaussian_with_mask, \
    sample_center_gravity_zero_gaussian_with_mask
from training.dataset.qm9 import get_cfg, get_dataset_info


def get_forward_rate(rate_function_name, max_problem_dim, rate_cut_t):
    
    if rate_function_name == 'step':
        return StepForwardRate(max_problem_dim, rate_cut_t)
    elif rate_function_name == 'const':
        return ConstForwardRate(max_problem_dim, None)
    else:
        raise ValueError(rate_function_name)


def get_noise_schedule(noise_schedule_name, max_problem_dim, vp_sde_beta_min, vp_sde_beta_max): 
    if noise_schedule_name == 'vp_sde':
        # DDPM schedule is beta_min=0.1, beta_max=20
        return VP_SDE(max_problem_dim, vp_sde_beta_min, vp_sde_beta_max)
    else:
        raise ValueError(noise_schedule_name)

@persistence.persistent_class
class JumpLossFinalDim:
    def __init__(self, structure, rate_function_name, min_t, vp_sde_beta_min, vp_sde_beta_max, rate_cut_t, loss_type,
                 x0_logit_ce_loss_weight, rate_loss_weight, score_loss_weight, auto_loss_weight,
                 noise_schedule_name,
                 mean_or_sum_over_dim,
                 nearest_atom_pred, nearest_atom_loss_weight):
        self.structure = structure
        self.min_t = min_t
        self.loss_type = loss_type

        self.forward_rate = get_forward_rate(rate_function_name,
            self.structure.graphical_structure.max_problem_dim,
            rate_cut_t)

        self.noise_schedule = get_noise_schedule(noise_schedule_name,
            self.structure.graphical_structure.max_problem_dim, vp_sde_beta_min,
            vp_sde_beta_max)

        self.x0_logit_ce_loss_weight = x0_logit_ce_loss_weight
        self.rate_loss_weight = rate_loss_weight
        self.score_loss_weight = score_loss_weight
        self.auto_loss_weight = auto_loss_weight

        self.mean_or_sum_over_dim = mean_or_sum_over_dim

        self.nearest_atom_pred = nearest_atom_pred
        self.nearest_atom_loss_weight = nearest_atom_loss_weight

    def __call__(self, net, st_batch):

        try:  # in case we're using a DataParallel model
            net.module.noise_schedule = self.noise_schedule
            net.module.model.noise_schedule = self.noise_schedule
        except: # in case we're not using a DataParallel model
            net.noise_schedule = self.noise_schedule
            net.model.noise_schedule = self.noise_schedule

        # inputs network and structured data batch

        B = st_batch.B
        device = st_batch.get_device()
        x0_dims = st_batch.get_dims()

        assert (self.structure.exist == 1).all()


        ts = self.min_t + (1-self.min_t) * torch.rand((B,)) # (B,)

        # delete some dimensions
        dims_xt = self.forward_rate.get_dims_at_t(
            start_dims=st_batch.get_dims(),
            ts =ts
        ).int() # (B,)
        st_batch.delete_dims(new_dims=dims_xt)

        st_batch.gs.adjust_st_batch(st_batch)


        x, y = st_batch.get_flat_lats_and_obs()

        mean, std = self.noise_schedule.get_p0t_stats(st_batch, ts.to(device))
        noise = torch.randn_like(mean)
        noise_st_batch = StructuredDataBatch.create_copy(st_batch)
        noise_st_batch.set_flat_lats(noise)
        noise_st_batch.delete_dims(new_dims=dims_xt)
        noise_st_batch.gs.adjust_st_batch(noise_st_batch)
        noise = noise_st_batch.get_flat_lats()
        xt = mean + std * noise

        st_batch.set_flat_lats(xt)

        # make sure all masks are still correct
        st_batch.delete_dims(new_dims=dims_xt)
        # adjust
        st_batch.gs.adjust_st_batch(st_batch)


        # first network pass
        to_predict = {'eps': 'eps', 'x0': 'x0', 'edm': 'x0'}[self.loss_type]
        if self.nearest_atom_pred:
            D_xt, rate_xt, dummy_mean_std, x0_dim_logits, _ = net(
                st_batch, ts=ts.to(device), forward_rate=self.forward_rate,
                predict=to_predict, nearest_atom=torch.zeros((B,), device=device).long()
            )
        else:
            D_xt, rate_xt, dummy_mean_std, x0_dim_logits = net(st_batch, ts=ts.to(device), forward_rate=self.forward_rate, predict=to_predict)
        assert rate_xt.shape == (B, 1)

        assert x0_dim_logits.shape == (B, st_batch.gs.max_problem_dim)
        # x0_dims (B,) 1 to max_dim
        # so need to subtract 1 for ce loss
        ce_loss = F.cross_entropy(x0_dim_logits, x0_dims.to(device)-1, reduction='none')
        assert ce_loss.shape == (B,)


        D_xt_mask = st_batch.get_mask(B, include_obs=False, include_onehot_channels=True).to(st_batch.get_device())
        D_xt = D_xt * D_xt_mask   # zero out for dimensions that don't exist

        # second network pass

        # remove the final dimension
        delxt_st_batch = StructuredDataBatch.create_copy(st_batch)
        delxt_st_batch.delete_one_dim()

        if self.nearest_atom_pred:
            nearest_atom = st_batch.gs.get_nearest_atom(st_batch, delxt_st_batch)
            assert nearest_atom.shape == (B,)

        adjust_val = delxt_st_batch.gs.adjust_st_batch(delxt_st_batch)

        if self.rate_loss_weight > 0 or self.score_loss_weight > 0:
            if self.nearest_atom_pred:
                _, rate_delxt, mean_std, _, near_atom_logits = net(
                    delxt_st_batch, ts=ts.to(device), forward_rate=self.forward_rate,
                    predict=to_predict, nearest_atom=nearest_atom
                )
            else:
                _, rate_delxt, mean_std, _ = net(delxt_st_batch, ts=ts.to(device), forward_rate=self.forward_rate, predict=to_predict)
        else:
            # dummy assignments to avoid errors
            rate_delxt = rate_xt
            mean_std = dummy_mean_std

        target = {'eps': noise, 'x0': x}[to_predict]
        score_loss = 0.5 * D_xt_mask * ( ( D_xt - target )**2)
        if self.loss_type == 'edm':
            vp_sigma = std
            vp_alpha = torch.sqrt(1-vp_sigma**2)
            ve_sigma = vp_sigma / vp_alpha
            weights = (ve_sigma**2 + 1) / ve_sigma**2
            score_loss = score_loss * weights
        
        f_rate_vs_t = self.forward_rate.get_rate(dims_xt, ts).to(device) # (B,)

        dims_xt = dims_xt.to(device)
        # if dims[idx] == max_dim then only do the log bit
        # if dims[idx] == 1 then only do the non-log bit
        rate_loss = \
            (dims_xt < st_batch.gs.max_problem_dim) * rate_xt[:, 0] \
            - (dims_xt >1) * f_rate_vs_t * torch.log(rate_delxt[:, 0] + 1e-12)
        assert rate_loss.shape == (B,)

        final_dim_mask = st_batch.get_next_dim_deleted_mask(B, include_onehot_channels=True, include_obs=False).to(x.device)

        # check mean and std are correctly masked
        # check that masked values are zero and that there's at least one big
        # non-masked value
        assert (mean_std[0] * (1 - final_dim_mask)).abs().max() < 1e-5
        assert (mean_std[1] * (1 - final_dim_mask)).abs().max() < 1e-5
        assert mean_std[0].abs().max() > 1e-5
        assert mean_std[1].abs().max() > 1e-5

        mean = mean_std[0]
        std = torch.nn.functional.softplus(mean_std[1])

        if sum(dims_xt > 1) > 0:
            auto_target = st_batch.gs.get_auto_target(st_batch, adjust_val)
            assert auto_target.shape == mean.shape
            auto_loss = - f_rate_vs_t * (dims_xt > 1) * \
                torch.sum(final_dim_mask * (-torch.log(std) - 0.5 * (1/(std**2)) * (auto_target - mean)**2), dim=1) 
            assert auto_loss.shape == (B,)
        else:
            auto_target = mean
            auto_loss = torch.zeros_like(rate_loss)

        if self.nearest_atom_pred:
            nearest_atom_loss = (dims_xt > 1) * F.cross_entropy(near_atom_logits, nearest_atom, reduction='none')
            assert nearest_atom_loss.shape == (B,)
        else:
            nearest_atom_loss = torch.zeros_like(rate_loss)

        # return loss unaveraged
        loss = self.score_loss_weight * score_loss + \
            self.rate_loss_weight * (1/x.shape[1]) * rate_loss.view(B, 1) + \
            self.auto_loss_weight * (1/x.shape[1]) * auto_loss.view(B, 1) + \
            self.x0_logit_ce_loss_weight * (1/x.shape[1]) * ce_loss.view(B,1) + \
            self.nearest_atom_loss_weight * (1/x.shape[1]) * nearest_atom_loss.view(B,1)


        loss_components = {'score_loss': score_loss,
                           'rate_loss': rate_loss.view(B, 1),
                           'auto_loss': auto_loss.view(B, 1),
                           'ce_loss': ce_loss.view(B,1),
                           'nearest_atom_loss': nearest_atom_loss.view(B,1),
                           'max_rate_xt': rate_xt.max(),
                           'min_rate_delxt': rate_delxt.min(),
                           'min_auto_std': std.min(),
                           'max_auto_L2': ((auto_target - mean)**2).max(),
        }

        if self.mean_or_sum_over_dim == 'mean':
            loss = (1/D_xt.shape[1]) * loss
        elif self.mean_or_sum_over_dim == 'sum':
            loss = loss
        else:
            raise ValueError(self.mean_or_sum_over_dim)

        return loss, loss_components

JumpLossFinalDim_to_kwargs = {
    JumpLossFinalDim: set([
        ('rate_function_name', 'click.Choice([\'const\', \'quartic\', \'step\', \'cosine\', \'dimsteprate\', \'pendulum\', \'triangle\', \'twostep\'])', 'const'),
        ('min_t', 'float', 0.001),
        ('vp_sde_beta_min', 'float', 0.1),
        ('vp_sde_beta_max', 'float', 20.0),
        ('rate_cut_t', 'float', 0.5),
        ('loss_type', 'str', 'eps'),
        ('x0_logit_ce_loss_weight', 'float', 0.1),
        ('rate_loss_weight', 'float', 1),
        ('score_loss_weight', 'float', 1),
        ('auto_loss_weight', 'float', 1),
        ('nearest_atom_loss_weight', 'float', 1.0),
        ('noise_schedule_name', 'click.Choice([\'vp_sde\', \'nonisocosine\', \'polynomial\'])', 'vp_sde'),
        ('mean_or_sum_over_dim', 'click.Choice([\'mean\', \'sum\'])', 'sum'),
        ('nearest_atom_pred', 'str2bool', 'False'),
    ]),
}

losses_to_kwargs = {
    l.__name__: kwargs for l, kwargs in it.chain(
        JumpLossFinalDim_to_kwargs.items(),
    )
}