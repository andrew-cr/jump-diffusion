import torch
from .egnn import EGNNMultiHeadJump, EGNNMultiHeadJump_to_kwargs
import itertools as it


#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

# @persistence.persistent_class
# class EDMPrecond(torch.nn.Module):
#     def __init__(self,
#         structure       = None,
#         use_fp16        = False,            # Execute the underlying model at FP16 precision?
#         sigma_min       = 0,                # Minimum supported noise level.
#         sigma_max       = float('inf'),     # Maximum supported noise level.
#         sigma_data      = 0.5,              # Expected standard deviation of the training data.
#         model_type      = 'DhariwalUNet',   # Class name of the underlying model.
#         pred_x0         = None,
#         noise_mult      = None,
#         **model_kwargs,                     # Keyword arguments for the underlying model.
#     ):
#         super().__init__()
#         self.structure = structure
#         self.use_fp16 = use_fp16
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.sigma_data = sigma_data
#         self.pred_x0 = StructuredArgument(pred_x0, structure=structure, dtype=torch.uint8)
#         self.noise_mult = StructuredArgument(noise_mult, structure=structure, dtype=torch.float32)
#         self.model = globals()[model_type](**model_kwargs, structure=structure)

#     def forward(self, x, y, sigma, force_fp32=False, **model_kwargs):
#         x = x.to(torch.float32)
#         sigma = sigma.to(torch.float32).reshape(-1, *[1]*(x.ndim-1))
#         dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

#         sigma_full = sigma * self.noise_mult.lats.to(sigma.device)
#         c_skip = self.sigma_data ** 2 / (sigma_full ** 2 + self.sigma_data ** 2)
#         c_out = sigma_full * self.sigma_data / (sigma_full ** 2 + self.sigma_data ** 2).sqrt()
#         c_in = 1 / (self.sigma_data ** 2 + sigma_full ** 2).sqrt()
#         c_noise = sigma.log() / 4

#         # modify c_skip, c_out to make F_x predict x0 for onehot variables
#         # D_x predicts x0 - so if F_x is x0 then c_skip should be zero and c_out should be one
#         pred_x0_lats = self.pred_x0.lats
#         if pred_x0_lats.bool().any():
#             pred_x0_lats = pred_x0_lats.to(x.device).to(torch.float32)
#             c_skip = c_skip * (1 - pred_x0_lats)
#             c_out = c_out * (1 - pred_x0_lats) + pred_x0_lats

#         F_x = self.model((c_in * x).to(dtype), y, c_noise.flatten(), **model_kwargs)
#         returned_tuple = isinstance(F_x, tuple)
#         if returned_tuple:
#             F_x, *others = F_x
#         assert F_x.dtype == dtype
#         D_x = c_skip * x + c_out * F_x.to(torch.float32)
#         return (D_x, *others) if returned_tuple else D_x

#     def round_sigma(self, sigma):
#         return torch.as_tensor(sigma)


# -------------------------
class EpsilonPrecond(torch.nn.Module):
    def __init__(self, structure, model_type,
                 use_fp16=-1, # not used but needed for compatibility
                 **model_kwargs):
        super().__init__()
        self.structure = structure
        self.model = globals()[model_type](**model_kwargs, structure=structure)

    def forward(self, st_batch, ts, predict='eps', **model_kwargs):
        xt = st_batch.get_flat_lats()  # TODO mode to relevant if statement below
        eps, *others = self.model(st_batch, ts, **model_kwargs)
        if predict == 'eps':
            return eps, *others
        elif predict == 'x0':
            x0 = self.noise_schedule.predict_x0_from_xt(xt, eps, ts)
            return x0, *others
        else:
            raise NotImplementedError(f'predict {predict} not implemented')


class X0Precond(EpsilonPrecond):
    def forward(self, st_batch, ts, predict='x0', **model_kwargs):
        xt = st_batch.get_flat_lats()  # TODO mode to relevant if statement below
        x0, *others = super().forward(st_batch, ts, predict='eps', **model_kwargs)  # predict='eps' tells EpsilonPrecond to return raw network output
        if predict == 'x0':
            return x0, *others
        elif predict == 'eps':
            eps = self.noise_schedule.predict_eps_from_x0_xt(st_batch, x0, ts)
            return eps, *others


class EDMPrecond(EpsilonPrecond):
    def forward(self, st_batch, ts, predict='x0', **model_kwargs):
        xt = st_batch.get_flat_lats()  # TODO mode to relevant if statement below
        thing, *others = super().forward(st_batch, ts, predict='eps', **model_kwargs)  # predict='eps' tells EpsilonPrecond to return raw network output
        x0 = self.get_x0_from_thing(thing, xt, ts)
        if predict == 'x0':
            return x0, *others
        elif predict == 'eps':
            # eps = self.noise_schedule.predict_eps_from_x0_xt(xt, x0, ts)  # not numberically stable when t is near 0
            eps = self.get_eps_from_thing(thing, xt, ts)
            return eps, *others

    def get_x0_from_thing(self, thing, xt, ts):
        """
        thing being network output
        based on EDM loss, but assume sigma_data is 1
        """
        sigma_vp = self.noise_schedule.get_sigma(ts).view(-1, 1)
        alpha_vp = torch.sqrt(1-sigma_vp**2)
        sigma_ve = sigma_vp / alpha_vp
        c_skip = 1 / torch.sqrt(sigma_ve**2 + 1)
        c_out = sigma_ve / torch.sqrt(sigma_ve**2 + 1)
        return c_skip * xt + c_out * thing

    def get_eps_from_thing(self, thing, xt, ts):
        sigma_vp = self.noise_schedule.get_sigma(ts).view(-1, 1)
        alpha_vp = torch.sqrt(1-sigma_vp**2)
        sigma_ve = sigma_vp / alpha_vp
        c_skip = 1 / torch.sqrt(sigma_ve**2 + 1)
        c_out = sigma_ve / torch.sqrt(sigma_ve**2 + 1)
        ve_xt = xt * torch.sqrt(1+sigma_ve**2)
        # x0 = c_skip * xt + c_out * thing
        # eps = (ve_xt - x0) / sigma_ve
        ### - rewrite to be more stable -------------------
        eps_times_sigma = (1-c_skip) * ve_xt - c_out * thing
        return eps_times_sigma / sigma_ve

class NonePrecond(torch.nn.Module):
    def __init__(self, structure, model_type, use_fp16=-1, **model_kwargs):
        super().__init__()
        self.structure = structure
        self.model = globals()[model_type](**model_kwargs, structure=structure)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

networks_to_kwargs = {
    l.__name__: kwargs for l, kwargs in it.chain(
        EGNNMultiHeadJump_to_kwargs.items(),
    )
}