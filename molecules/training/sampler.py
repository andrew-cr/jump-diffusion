import numpy as np
import torch
import torch.nn.functional as F
import math
from training.structure import StructuredDataBatch
from training.networks.egnn import EGNNMultiHeadJump
import itertools as it

from training.dataset.qm9 import get_cfg, get_dataset_info
from training.egnn_utils import sample_center_gravity_zero_gaussian_with_mask, \
    sample_gaussian_with_mask, assert_correctly_masked, assert_mean_zero_with_mask, \
    remove_mean_with_mask

from training.diffusion_utils import get_rate_using_x0_pred


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.rand(size[1:], generator=gen, **kwargs) for gen in self.generators])
    
    def poisson(self, input):
        assert input.shape[0] == len(self.generators)
        return torch.stack([torch.poisson(input[i], generator=gen) for i, gen in enumerate(self.generators)])
    
    def multinomial(self, input, **kwargs):
        assert input.shape[0] == len(self.generators)
        return torch.stack([torch.multinomial(input[i, ...], generator=gen, **kwargs) for i, gen in enumerate(self.generators)])



class JumpSampler():
    def __init__(self, structure, dt, corrector_steps, corrector_snr,
                 corrector_start_time, corrector_finish_time,
                 do_conditioning, condition_type, condition_sweep_idx, condition_sweep_path, 
                 guidance_weight, do_jump_corrector, sample_near_atom,
                 dt_schedule, dt_schedule_h, dt_schedule_l,
                 dt_schedule_tc, no_noise_final_step):
        self.structure = structure
        self.dt = dt
        self.corrector_steps = corrector_steps # 0 for no corrector
        self.corrector_snr = corrector_snr # scaling for corrector step size
        self.corrector_start_time = corrector_start_time
        self.corrector_finish_time = corrector_finish_time

        self.do_conditioning = do_conditioning
        self.condition_type = condition_type
        self.condition_sweep_idx = condition_sweep_idx
        self.condition_sweep_path = condition_sweep_path

        self.guidance_weight = guidance_weight
        self.do_jump_corrector = do_jump_corrector
        self.sample_near_atom = sample_near_atom

        self.dt_schedule = dt_schedule
        self.dt_schedule_h = dt_schedule_h
        self.dt_schedule_l = dt_schedule_l
        self.dt_schedule_tc = dt_schedule_tc

        self.no_noise_final_step = no_noise_final_step

    def get_dt(self, ts):
        if self.dt_schedule == 'uniform':
            return self.dt
        elif self.dt_schedule == 'C':
            h = self.dt_schedule_h
            l = self.dt_schedule_l
            tc = self.dt_schedule_tc
            return (ts > tc).long() * h + (ts <= tc).long() * l
        else:
            raise NotImplementedError(self.dt_schedule)

    def get_score(self, state_st_batch, net, loss, ts, dataset_obj, rnd):
        if not self.do_conditioning:
            if self.sample_near_atom:
                D_xt, rate_xt, mean_std, _, _ = net(state_st_batch, ts, nearest_atom=None, sample_nearest_atom=True, forward_rate=loss.forward_rate, predict='eps', rnd=rnd)
            else:
                D_xt, rate_xt, mean_std, _ = net(state_st_batch, ts, forward_rate=loss.forward_rate, predict='eps')

            mean, std = loss.noise_schedule.get_p0t_stats(state_st_batch, ts)

            score = -(1/torch.clamp(std, min=0.001)) * D_xt

            return score, rate_xt, mean_std
        else:
            flat_lats = state_st_batch.get_flat_lats().detach()
            flat_lats.requires_grad = True
            state_st_batch.set_flat_lats(flat_lats)
            batch = state_st_batch.B
            num_dims = state_st_batch.get_dims()

            if self.sample_near_atom:
                D_xt, rate_xt, mean_std, _, _ = net(state_st_batch, ts, nearest_atom=None, sample_nearest_atom=True, forward_rate=loss.forward_rate, predict='eps', rnd=rnd)
            else:
                D_xt, rate_xt, mean_std, _ = net(state_st_batch, ts, forward_rate=loss.forward_rate, predict='eps')

            x0_pred = loss.noise_schedule.predict_x0_from_xt(state_st_batch.get_flat_lats(), D_xt, ts)

            condition_st_batch, condition_mask = dataset_obj.condition_state(state_st_batch, self.condition_type, self.condition_sweep_idx, self.condition_sweep_path)
            condition_dims = condition_st_batch.get_dims()

            x0_pred_of_cond = condition_mask * x0_pred
            x0_pred_of_cond_st_batch = StructuredDataBatch.create_copy(state_st_batch)
            x0_pred_of_cond_st_batch.set_flat_lats(x0_pred_of_cond)
            x0_pred_of_cond_st_batch.set_dims(condition_dims)
            x0_pred_of_cond_st_batch.delete_dims(new_dims=condition_dims)
            x0_pred_of_cond_st_batch.gs.adjust_st_batch(x0_pred_of_cond_st_batch)
            x0_pred_of_cond = x0_pred_of_cond_st_batch.get_flat_lats()

            l2_error = torch.sum( condition_mask * (x0_pred_of_cond - condition_st_batch.get_flat_lats())**2, dim=1) # (B,)

            unit_st_batch = StructuredDataBatch.create_copy(state_st_batch)
            unit_st_batch.set_flat_lats(torch.ones_like(unit_st_batch.get_flat_lats()))
            alpha_t = loss.noise_schedule.get_p0t_stats(unit_st_batch, ts)[0][:, 0] # (B,)
            l2_error = -0.5 * self.guidance_weight * alpha_t * l2_error

            guidance_grad = torch.autograd.grad(l2_error, flat_lats,
                grad_outputs=torch.ones_like(l2_error),
                allow_unused=True)[0]
            # (B, 261)

            x0_pred_adjusted = condition_mask * condition_st_batch.get_flat_lats() + \
                (1-condition_mask) * (
                    x0_pred + guidance_grad
                )

            x0_with_condition_st_batch = StructuredDataBatch.create_copy(state_st_batch)
            x0_with_condition_st_batch.set_flat_lats(x0_pred_adjusted)
            x0_with_condition_st_batch.set_dims(num_dims)
            x0_with_condition_st_batch.delete_dims(new_dims=num_dims)
            x0_with_condition_st_batch.gs.adjust_st_batch(x0_with_condition_st_batch)
            x0_with_condition = x0_with_condition_st_batch.get_flat_lats()
            eps_pred = loss.noise_schedule.predict_eps_from_x0_xt(state_st_batch, x0_with_condition, ts)
            
            _, std = loss.noise_schedule.get_p0t_stats(state_st_batch, ts)

            score = (-1/torch.clamp(std, min=0.001)) * eps_pred
            return score, rate_xt, mean_std

    def sample(self, net, in_st_batch, loss, rnd, known_dims=None, dataset_obj=None):
        print('---------- jump sampler -------------')

        try:  # in case we're using a DataParallel model
            net.module.noise_schedule = loss.noise_schedule
            net.module.model.noise_schedule = loss.noise_schedule
        except: # in case we're not using a DataParallel model
            net.noise_schedule = loss.noise_schedule
            net.model.noise_schedule = loss.noise_schedule

        state_st_batch = StructuredDataBatch.create_copy(in_st_batch)
        max_problem_dim = state_st_batch.gs.max_problem_dim

        x0, y = state_st_batch.get_flat_lats_and_obs()

        B = x0.shape[0]

        xT = rnd.randn_like(x0) # (initialize at N(0, I))

        state_st_batch.set_flat_lats(xT)

        # start at dimension 1
        num_dims = torch.ones((B,)).long()

        state_st_batch.delete_dims(new_dims=num_dims)

        state_st_batch.gs.adjust_st_batch(state_st_batch)

        device=x0[0].device
        ts = torch.ones((B,), device=device)
        steps_since_added_dim = torch.inf * torch.ones((B,), device=device)

        finish_at = self.dt/2

        nfe = 0
        will_finish = False

        while True:

            if (ts - self.get_dt(ts)).clamp(min=finish_at/2).max() < finish_at:
                will_finish = True

            # corrector steps
            if ts.min() < self.corrector_start_time and ts.max() > self.corrector_finish_time:
                corrector_steps = self.corrector_steps
            else:
                corrector_steps = 0

            xt = state_st_batch.get_flat_lats()
            def set_unfinished_lats(xt):
                state_st_batch.set_flat_lats(xt * (1-is_finished) + state_st_batch.get_flat_lats() * is_finished)

            # implement corrector steps for after adding a dimension
            is_finished = (ts < finish_at).float().view(-1, 1)

            # diffusion bit
            beta_t = loss.noise_schedule.get_beta_t(ts) # (B, problem_dim)
            beta_t = state_st_batch.convert_problem_dim_to_tensor_dim(beta_t) # (B, tensor_dim)

            score, rate_xt, mean_std = self.get_score(state_st_batch, net, loss, ts, dataset_obj, rnd)
            nfe += 1

            mask = state_st_batch.get_mask(B=B, include_obs=False, include_onehot_channels=True).to(device)

            xt = (2 - torch.sqrt(1 - beta_t * self.dt)) * xt + \
                mask * beta_t * self.dt * score
            noise = rnd.randn_like(xt)
            noise_st_batch = StructuredDataBatch.create_copy(state_st_batch)
            noise_st_batch.set_flat_lats(noise)
            noise_st_batch.delete_dims(new_dims=num_dims)
            noise_st_batch.gs.adjust_st_batch(noise_st_batch)
            noise = noise_st_batch.get_flat_lats()

            if not(corrector_steps == 0 and self.no_noise_final_step and will_finish):
                xt = xt + mask * torch.sqrt(beta_t * self.dt) * noise

            set_unfinished_lats(xt)
            state_st_batch.gs.adjust_st_batch(state_st_batch)
            xt = state_st_batch.get_flat_lats()

            # jump bit
            rate_xt = rate_xt.squeeze(1)
            increase_mask = (rnd.rand((B,), device=device) < rate_xt * self.dt) * (num_dims.to(device) < max_problem_dim) # (B,)

            increase_mask = (1 - is_finished.view(-1)).bool() * increase_mask  # don't increase dimension after we've finished

            next_dims_mask = state_st_batch.get_next_dim_added_mask(B=B, include_onehot_channels=True, include_obs=False).to(device)
            mean = mean_std[0]
            std = torch.nn.functional.softplus(mean_std[1])
            new_values = next_dims_mask * (mean + rnd.randn_like(std) * std)
            xt[increase_mask, :] = xt[increase_mask, :] * (1-next_dims_mask[increase_mask, :]) + new_values[increase_mask, :]

            num_dims[increase_mask.to(num_dims.device)] = num_dims[increase_mask.to(num_dims.device)] + 1

            state_st_batch.set_dims(num_dims)
            set_unfinished_lats(xt.detach())
            state_st_batch.delete_dims(num_dims)
            state_st_batch.gs.adjust_st_batch(state_st_batch)
            xt = state_st_batch.get_flat_lats().detach()


            for corrector_idx in range(corrector_steps):
                set_unfinished_lats(xt)
                beta_tm1 = loss.noise_schedule.get_beta_t(ts-self.dt) # (B, problem_dim)
                beta_tm1 = state_st_batch.convert_problem_dim_to_tensor_dim(beta_tm1) # (B, tensor_dim)

                score, rate_xt, mean_std = self.get_score(state_st_batch, net, loss, ts-self.dt, dataset_obj, rnd)
                nfe += 1

                noise = rnd.randn_like(xt)
                noise_st_batch = StructuredDataBatch.create_copy(state_st_batch)
                noise_st_batch.set_flat_lats(noise)
                noise_st_batch.delete_dims(new_dims=num_dims)
                noise_st_batch.gs.adjust_st_batch(noise_st_batch)
                noise = noise_st_batch.get_flat_lats()
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                alpha = 1 - self.dt * beta_tm1
                step_size = (self.corrector_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                if corrector_idx == corrector_steps - 1 and self.no_noise_final_step and will_finish:
                    xt = xt + mask * (step_size * score)
                else:
                    xt = xt + mask * (step_size * score + torch.sqrt(2 * step_size) * noise)
                set_unfinished_lats(xt.detach())
                state_st_batch.gs.adjust_st_batch(state_st_batch)
                xt = state_st_batch.get_flat_lats().detach()


                # jump correction
                if self.do_jump_corrector:

                    rate_xt = rate_xt.squeeze(1)
                    increase_mask = (rnd.rand((B,), device=device) < rate_xt * self.dt) * (num_dims.to(device) < max_problem_dim) # (B,)
                    decrease_mask = (rnd.rand((B,), device=device) < loss.forward_rate.get_rate(None, ts-self.dt) * self.dt) * (num_dims.to(device) > 1) # (B,)

                    increase_mask = (1 - is_finished.view(-1)).bool() * increase_mask  # don't increase dimension after we've finished
                    decrease_mask = (1 - is_finished.view(-1)).bool() * decrease_mask  # don't decrease dimension after we've finished

                    next_dims_mask = state_st_batch.get_next_dim_added_mask(B=B, include_onehot_channels=True, include_obs=False).to(device)
                    mean = mean_std[0]
                    std = torch.nn.functional.softplus(mean_std[1])
                    new_values = next_dims_mask * (mean + rnd.randn_like(std) * std)
                    xt[increase_mask, :] = xt[increase_mask, :] * (1-next_dims_mask[increase_mask, :]) + new_values[increase_mask, :]

                    num_dims[increase_mask.to(num_dims.device)] = num_dims[increase_mask.to(num_dims.device)] + 1

                    state_st_batch.set_dims(num_dims)
                    set_unfinished_lats(xt.detach())

                    # now remove any atoms
                    num_dims[decrease_mask.to(num_dims.device)] = num_dims[decrease_mask.to(num_dims.device)] - 1
                    state_st_batch.set_dims(num_dims)
                    state_st_batch.delete_dims(new_dims=num_dims)

                    state_st_batch.gs.adjust_st_batch(state_st_batch)
                    xt = state_st_batch.get_flat_lats().detach()


            dt = self.get_dt(ts) 
            ts -= dt
            ts = ts.clamp(min=finish_at/2)  # don't make zero in case of numerical weirdness
            if ts.max() < finish_at:  # miss the last step, as it seems to improve RMSE...
                break

        print('-------- finish sampling ---------')
        print('nfe: ', nfe)

        return state_st_batch

JumpSampler_to_kwargs = {
    JumpSampler: set([
        ('dt', 'float', 0.001),
        ('corrector_steps', 'int', 0),
        ('corrector_snr', 'float', 0.1),
        ('corrector_start_time', 'float', 0.1),
        ('corrector_finish_time', 'float', 0.003),
        ('do_conditioning', 'str2bool', 'False'),
        ('condition_type', 'click.Choice([\'sweep\'])', 'sweep'),
        ('condition_sweep_idx', 'int', 0),
        ('condition_sweep_path', 'str', None),
        ('guidance_weight', 'float', 1.0),
        ('do_jump_corrector', 'str2bool', 'False'),
        ('sample_near_atom', 'str2bool', 'False'),
        ('dt_schedule', 'click.Choice([\'uniform\', \'C\'])', 'uniform'),
        ('dt_schedule_h', 'float', 0.05),
        ('dt_schedule_l', 'float', 0.001),
        ('dt_schedule_tc', 'float', 0.5),
        ('no_noise_final_step', 'str2bool', 'False'),
    ]),
}




samplers_to_kwargs = {
    l.__name__: kwargs for l, kwargs in it.chain(
        JumpSampler_to_kwargs.items(),
    )
}