#%%
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
from training.structure import StructuredArgument
from functools import partial
from math import log
import os
from einops import rearrange

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(
        self, structure, noise_mult, pred_x0, P_mean=-1.2, P_std=1.2, sigma_data=0.5, jump_diffusion=False, any_dimension_deletion=False, highly_nonisotropic=False, jump_per_index=False,
        just_jump_loss=False, dim_del_sigma_min=1, dim_del_sigma_max=10,
    ):
        self.structure = structure
        # self.noise_mult = StructuredArgument(noise_mult, structure=structure, dtype=torch.float32)
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.jump_diffusion = jump_diffusion
        self.just_jump_loss = just_jump_loss
        self.highly_nonisotropic = highly_nonisotropic
        self.dim_del_sigma_min = dim_del_sigma_min
        self.dim_del_sigma_max = dim_del_sigma_max
        if jump_diffusion:
            self.dim_deletion_process = DimDeletionProcess(
                P_mean, P_std, preaugmented_structure=structure, any_dimension_deletion=any_dimension_deletion, highly_nonisotropic=highly_nonisotropic, jump_per_index=jump_per_index,
                dim_del_sigma_min=dim_del_sigma_min, dim_del_sigma_max=dim_del_sigma_max,
            )
            self.preaugment_structure = structure
            self.structure = self.preaugment_structure.get_embedded_version(
                embed_func=lambda x: self.dim_deletion_process.prepare_batch(x, sigmas=torch.tensor([1.0]))[0],
            )
            self.jump_per_index = jump_per_index
        else:
            self.dim_deletion_process = None
            self.structure = structure
        self.pred_x0 = StructuredArgument(pred_x0, structure=self.structure, dtype=torch.uint8)

    def __call__(self, net, x, y, augment_labels):
        B = x.shape[0]
        assert self.highly_nonisotropic, 'Noise labels not properrly implemented'
        while True:
            # rejection sampling to ensure sigma is not too great
            if self.just_jump_loss and self.highly_nonisotropic:
                sigma = torch.zeros([B, 1], device=x.device)
            elif self.just_jump_loss and not self.highly_nonisotropic:
                rnd_uniform = torch.rand([B, 1], device=x.device)
                expanded_rnd_uniform = -0.01 + 1.02 * rnd_uniform
                _min, _max = log(self.dim_del_sigma_min), log(self.dim_del_sigma_max)
                logsigma = (_max - _min) * expanded_rnd_uniform + _min
                sigma = logsigma.exp()
            else:
                rnd_normal = torch.randn([B, 1], device=x.device)
                sigma = (rnd_normal * self.P_std + self.P_mean).exp()  # 95% interval between exp(-3.6) and exp(1.2)
            if (not hasattr(self, 'dim_deletion_process')) or (sigma.max() < self.dim_deletion_process.large_value):
                break

        if self.dim_deletion_process is not None:
            data = self.preaugment_structure.unflatten_batch(x, y, pad_marg=True)
            data, sigma_per_dim, num_dimensions_clear = self.dim_deletion_process.prepare_batch(data, sigma)
            x_target, y = self.structure.flatten_batch(data, contains_marg=True)
            x, _ = self.preaugment_structure.flatten_batch(data[:-2], contains_marg=True)
            dim_mask = tuple(t.unsqueeze(0) for t in self.structure.dim_handler.which_dim_mask(device=x.device))
            dim_mask = self.structure.flatten_latents(dim_mask, contains_marg=True).squeeze(0)
            sigma_full = torch.stack([sigma_per_dim[i][dim_mask] for i in range(B)])  # B x D
            n_extra_dims = self.structure.latent_dim - self.preaugment_structure.latent_dim
            extra_sigmas = torch.ones_like(sigma_full[:, :1]).expand(-1, n_extra_dims)  # value is arbitrary
            sigma_full = torch.cat([sigma_full, extra_sigmas], dim=1)
            extra_dummy_x = extra_sigmas
            x = torch.cat([x, extra_dummy_x], dim=1)
        else:
            x_target = x
            sigma_full = sigma

        n = torch.randn_like(x) * sigma_full
        D_xn = net(x+n, y=y, sigma_full=sigma_full, augment_labels=augment_labels)

        # only use MSE for score loss: ------------------------------
        score_dim = self.preaugment_structure.latent_dim
        x0_pred_dim = self.structure.dim_handler.max_dim
        if self.jump_diffusion and self.jump_per_index:
            index_pred_dim = (x0_pred_dim + 1) * (x0_pred_dim + 1)
        else:
            index_pred_dim = x0_pred_dim + 1
        assert x_target.shape[1] == score_dim + x0_pred_dim + index_pred_dim
        index_target = x_target[:, score_dim + x0_pred_dim:]
        x0_target = x_target[:, score_dim:score_dim + x0_pred_dim]
        x_target = x_target[:, :score_dim]
        x0_pred = D_xn[:, score_dim:score_dim + x0_pred_dim]
        index_pred = D_xn[:, score_dim + x0_pred_dim:]
        D_xn = D_xn[:, :score_dim]
        # -----------------------------------------------------------

        # print('constructing weight', sigma_full.min().item(), sigma_full.max().item(), self.sigma_data)
        weight = (sigma_full ** 2 + self.sigma_data ** 2) / (sigma_full * self.sigma_data) ** 2
        weight[sigma_full == 0] = 0
        pred_x0_lats = self.pred_x0.lats.to(x.device).view(1, -1)
        if pred_x0_lats.bool().any():
            # weight computed above is ~1/sigma**2 for whichever sigma is smallest. For small sigma_full,
            # this means weight is very large. When predicting x0 for onehots, we probably don't want this
            # scaling, so set all weights for onehots to 1.
            weight = weight * (1-pred_x0_lats) + torch.ones_like(weight) * pred_x0_lats
        weight = weight[:, :score_dim]
        loss = weight * ((D_xn - x_target) ** 2)
        # print('nans in xtarget:', x_target.isnan().sum().item())
        # print('infs in xtarget:', x_target.isinf().sum().item())
        # print('nans in D_xn:', D_xn.isnan().sum().item())
        # print('infs in D_xn:', D_xn.isinf().sum().item())
        # print('nans in weight:', weight.isnan().sum().item())
        # print('infs in weight:', weight.isinf().sum().item())
        # print('nans in loss:', loss.isnan().sum().item())
        # print('infs in loss:', loss.isinf().sum().item())
        # print('nans in score bit of loss:', loss[:, :score_dim].isnan().sum().item())
        # print('nans in x0 bit of loss:', loss[:, score_dim:score_dim + x0_pred_dim].isnan().sum().item())
        # print('nans in index bit of loss:', loss[:, score_dim + x0_pred_dim:].isnan().sum().item())
        # loss[sigma_full[:, :score_dim] == 0] = 0  # avoid nans when sigma is 0
        # print('filtered')
        # print('nans in loss:', loss.isnan().sum().item())
        # print('infs in loss:', loss.isinf().sum().item())
        # print()
        if self.dim_deletion_process is not None:
            data_shaped_loss_mask = self.preaugment_structure.dim_handler.batched_mask(datas=data, device=x.device)
            # data_shaped_loss_mask += (torch.ones_like(data[-2]), torch.ones_like(data[-1]),)  # add ones for dim-deletion stuff
            loss_mask = self.structure.flatten_latents(data_shaped_loss_mask, contains_marg=True)
            loss = loss * loss_mask


            if self.jump_per_index:

                max_dim = self.structure.dim_handler.max_dim
                _index_target = index_target.view(B, max_dim+1, max_dim+1)
                index_target = _index_target[:, :, :-1]
                actual_index_target = _index_target[:, :, -1]
                _index_pred = index_pred.view(B, max_dim+1, max_dim+1)
                index_pred = _index_pred[:, :, :-1]
                actual_index_pred = _index_pred[:, :, -1]
                # print('pre softmax index_pred', index_pred)
                log_index_pred = F.log_softmax(index_pred, dim=2)
                # print('log_index_pred', log_index_pred)
                log_likelihood = log_index_pred * index_target
                log_likelihood[torch.isinf(index_pred) * (index_target == 0)] = 0
                index_pred_loss = -log_likelihood.view(B, max_dim+1, max_dim).sum(dim=2, keepdim=True)

                log_actual_index_pred = F.log_softmax(actual_index_pred, dim=1)
                actual_index_target = actual_index_target / actual_index_target.sum(dim=1, keepdim=True).clamp(min=1)
                actual_log_likelihood = log_actual_index_pred * actual_index_target
                actual_index_pred_loss = -actual_log_likelihood.view(B, max_dim+1, 1)

                index_pred_loss = 1 / (max_dim+1) * (index_pred_loss + actual_index_pred_loss)
                index_pred_loss = index_pred_loss.expand(B, max_dim+1, max_dim+1).reshape(B, (max_dim+1)**2)

                # print()
                if hasattr(self, 'plot_path') and not os.path.exists(self.plot_path):
                    self.plot_predictions(log_index_pred.exp(), log_actual_index_pred.exp(), index_target, self.plot_path)

                # TODO also do cross-entropy loss for x0 prediction, and index pred loss
                x0_pred = F.log_softmax(x0_pred, dim=1)
                x0_pred_loss = F.cross_entropy(x0_pred, x0_target, reduction='none')
                # expected to be same shape as x0_pred, so spread it out
                x0_pred_loss = torch.ones_like(x0_pred) * x0_pred_loss.unsqueeze(1) / x0_pred.shape[1]
                # mask loss if timestep is small enough that it is clear how many dims x0 had
                x0_pred_loss = x0_pred_loss * (1 - num_dimensions_clear.unsqueeze(1))
                x0_pred_loss = x0_pred_loss.view(B, max_dim)

            else:
                # do cross-entropy loss for x0 prediction
                x0_pred = F.log_softmax(x0_pred, dim=1)
                x0_pred_loss = F.cross_entropy(x0_pred, x0_target, reduction='none')
                # expected to be same shape as x0_pred, so spread it out
                x0_pred_loss = torch.ones_like(x0_pred) * x0_pred_loss.unsqueeze(1) / x0_pred.shape[1]
                # mask loss if timestep is small enough that it is clear how many dims x0 had
                x0_pred_loss = x0_pred_loss * (1 - num_dimensions_clear.unsqueeze(1))

                # do cross-entropy loss for index prediction
                index_pred = F.log_softmax(index_pred, dim=1)
                # sometimes index_target is all zero (if no dims were deleted) - this is naturally handled
                # by cross-entropy loss
                # index_pred_loss = F.cross_entropy(index_pred, index_target, reduction='none')
                # # expected to be same shape as index_pred, so spread it out
                # index_pred_loss = torch.ones_like(index_pred) * index_pred_loss.unsqueeze(1) / index_pred.shape[1]
                index_pred_log_likelihood = index_pred * index_target
                index_pred_log_likelihood[torch.isinf(index_pred) * (index_target == 0)] = 0.  # avoid nans where we predict 0 and target prob is 0
                index_pred_loss = -index_pred_log_likelihood

            loss = torch.cat([loss, x0_pred_loss, index_pred_loss], dim=1)
        return loss

    def plot_predictions(self, x0_pred, index_pred, index_target, save_path):
        import matplotlib.pyplot as plt
        B, T, D = x0_pred.shape
        fig, axes = plt.subplots(B, T, figsize=(T, B))
        # print('PLOTTING INDEX PRED')
        # print('index_pred', index_pred)
        for b in range(B):
            for t in range(T):
                if not index_target[b, t].any():
                    continue
                axes[b, t].bar(list(range(D)), x0_pred[b, t].detach().cpu().numpy())
                max_ = x0_pred[b, t].max().item()
                index_target_val = index_target[b, t].argmax().item()
                axes[b, t].scatter(index_target_val, max_, color='red', marker='x')
                axes[b, t].bar([T], index_pred[b, t].detach().cpu().numpy())
        plt.savefig(save_path, bbox_inches='tight')

"""
precondition loss function by:
- adding latent variable describing number of dimensions
- deleting dimensions
- adding latent variable describing where dimensions were deleted

also, good to mask loss function based on which dimensions were deleted
"""

@persistence.persistent_class
class DimDeletionProcess():
    def __init__(self, P_mean, P_std, preaugmented_structure, large_value=100, any_dimension_deletion=False, highly_nonisotropic=False, jump_per_index=False,
                 dim_del_sigma_min=1, dim_del_sigma_max=10,
    ):
        # TODO there will be an error if sigma is sampled > large_value ?
        self.P_mean = P_mean
        self.P_std = P_std
        self.preaugmented_structure = preaugmented_structure
        self.max_dim = preaugmented_structure.dim_handler.max_dim
        self.any_dimension_deletion = any_dimension_deletion
        self.highly_nonisotropic = highly_nonisotropic
        self.large_value = large_value
        self.jump_per_index = jump_per_index
        # s_large = P_mean + P_std + 2  # 0, exp(0) = 1
        # s_small = P_mean - P_std + 2 # -2.4, exp(-2.4) = 0.09
        if not highly_nonisotropic:
            s_large = log(dim_del_sigma_max)  # log(10) = 2.3
            s_small = log(dim_del_sigma_min)  # log(1) = 0
            # P_mean, P_std = -0.6, 1.8
            # assert s_large < self.P_mean + 2 * self.P_std  # 2.3 < 3
            # assert s_small > self.P_mean - 2 * self.P_std  # 0 > -4.2
            self.deletion_sigmas = torch.linspace(s_large, s_small, self.max_dim-1).exp()

        # sigma = 100 -> 10 -> 1 -> 0

    def get_sigmas(self, sigma):
        self.deletion_sigmas = self.deletion_sigmas.to(sigma.device)
        is_deleted = (sigma > self.deletion_sigmas).float()
        near_threshold = self.deletion_sigmas/2
        near_deleted = (sigma > (near_threshold)).float() * (sigma < self.deletion_sigmas).float()
        # interp_slope = (self.large_value - near_threshold) / (self.deletion_sigmas - near_threshold)

        # interpolate from (near_threshold, near_threshold) to (self.deletion_sigmas, self.large_value)
        y_diff = self.large_value - near_threshold
        x_diff = self.deletion_sigmas - near_threshold
        y_intercept = near_threshold
        x_start = near_threshold
        alpha = (sigma - x_start) / x_diff
        transformed_alpha = alpha ** 5
        interp = y_intercept + transformed_alpha * y_diff
        full_sigma = is_deleted * self.large_value + near_deleted * interp + (1-is_deleted-near_deleted) * sigma
        full_sigma = torch.cat([torch.tensor([sigma], device=full_sigma.device), full_sigma])
        return full_sigma

    def get_highly_nonisotropic_sigmas(self, sigma, ndims=None, maxdims=None):
        """
        sigma is zero except when near threshold, where it is linearly interpolated to large_value
        """
        # sample number of dimensions. return sigma for that dimension
        if ndims is None:
            maxdims = maxdims if maxdims is not None else self.max_dim
            ndims = torch.randint(1, int(maxdims), size=()).item()
        full_sigma = torch.zeros(self.max_dim).to(sigma.device)
        full_sigma[:ndims-1] = 0.
        full_sigma[ndims-1:ndims] = sigma
        full_sigma[ndims:] = self.large_value
        return full_sigma

    def prepare_batch(self, data, sigmas):
        # get number of dims before deletions
        B = data[0].shape[0]
        x0_dims = self.preaugmented_structure.dim_handler.count_dims(data)
        x0_dims_onehot = torch.zeros([B, self.max_dim], device=sigmas.device)  # has size 30 (ignore that there is never only one dim in x0)
        for i, dims in enumerate(x0_dims):
            try:
                x0_dims_onehot[i, int(dims)-1] = 1  # has size 30 although 0th dim is always 0
            except IndexError:
                print('IndexError. This may occur durinig initialization. if it occurs multiple times, things are bad.')
                pass
        
        # delete dimensions
        if self.highly_nonisotropic:
            sigma_per_dim = torch.stack([self.get_highly_nonisotropic_sigmas(s, maxdims=x0_dim) for s, x0_dim in zip(sigmas, x0_dims)])
        else:
            sigma_per_dim = torch.stack([self.get_sigmas(s) for s in sigmas])
        num_dims_not_deleted = (sigma_per_dim < self.large_value).float().sum(dim=1)
        for b in range(B):
            # permute which dimensions are deleted, among those that exist
            exist_dims = int(x0_dims[b].item())
            if self.any_dimension_deletion:
                sigma_per_dim[b, :exist_dims] = sigma_per_dim[b, :exist_dims][torch.randperm(exist_dims)]
            sigma_per_dim[b, exist_dims:] = self.large_value
        exist = (sigma_per_dim < self.large_value).float()  # should have size 30
        exist_mask = self.preaugmented_structure.dim_handler.batched_mask(exist, device=sigmas.device)
        data = [d*e + (-1)*(1-e) for d, e in zip(data, exist_mask)]  # actually delete dimensions

        num_dimensions_clear = (num_dims_not_deleted > x0_dims).float()

        # compute number of deletions between each non-deleted item
        deletions_between = []
        for b in range(B):
            dels = 0
            db = []
            exist_dims = int(x0_dims[b].item())
            for e in exist[b, :exist_dims]:
                if e:
                    db.append(dels)
                    dels = 0
                else:
                    dels += 1
            db.append(dels)
            db.extend([0] * (self.max_dim + 1 - len(db)))
            db = torch.tensor(db).float()
            if not self.jump_per_index:
                db /= db.sum().clamp(min=1)
            deletions_between.append(db)  # this doesn't take into account that there may have been less than the max number of dimensions to begin with
        deletions_between = torch.stack(deletions_between).to(sigmas.device)

        if self.jump_per_index:
            # make deletions_between a big onehot prediction target
            new_deletions_between = torch.zeros([B, self.max_dim + 1, self.max_dim], device=sigmas.device)

            for b in range(B):
                for i, d in enumerate(deletions_between[b]):
                    n_dims_exist = int(sum(exist[b, :int(x0_dims[b])]).item())
                    if i <= n_dims_exist:
                        new_deletions_between[b, i, int(d.item())] = 1

            deletions_between = torch.cat([new_deletions_between, rearrange(deletions_between, 'b t -> b t 1')], dim=2)

        # augment data and return
        return (*data, x0_dims_onehot, deletions_between), sigma_per_dim, num_dimensions_clear
