import pickle
from training.networks.fdm import WrappedUNetVideoModel, VideoJumpPredictor, fdm_name_to_kwargs

arch_names_to_kwargs = fdm_name_to_kwargs

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        structure       = None,
        embedder        = None,
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        pred_x0         = None,
        noise_mult      = None,
        pretrained_weights = None,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        print('Initializing EDMPrecond')
        super().__init__()
        self.structure = structure
        self.embedder = embedder  # only used externally, but we tuck it in here to manage its params
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.pred_x0 = StructuredArgument(pred_x0, structure=structure, dtype=torch.uint8)
        # self.noise_mult = StructuredArgument(noise_mult, structure=structure, dtype=torch.float32)
        self.model = globals()[model_type](**model_kwargs, structure=structure)
        if pretrained_weights is not None:
            print(f'Loading pretrained weights from {pretrained_weights}')
            pretrained = pickle.load(open(pretrained_weights, 'rb'))['ema']
            pretrained_state_dict = self.model.edit_pretrained_state_dict(pretrained.state_dict())
            self.load_state_dict(pretrained_state_dict)
        try:
            self.model.add_control()
            print('Added control to network.')
        except AttributeError:
            pass
            

    def forward(self, x, y, sigma_full, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, *[1]*(x.ndim-1))
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # sigma_full = sigma * self.noise_mult.lats.to(sigma.device)
        c_skip = self.sigma_data ** 2 / (sigma_full ** 2 + self.sigma_data ** 2)
        c_out = sigma_full * self.sigma_data / (sigma_full ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_full ** 2).sqrt()
        truncated_sigma_full = sigma_full.clamp(min=1e-5)  # otherwise gives nans when some variables are fully denoised
        c_noise = truncated_sigma_full.log() / 4  # NOTE may have too many dimensions for architectures other than FDM
        data = self.structure.unflatten_latents(c_noise)
        c_noise = data[0][:, :, 0, 0, 0]  # a little hardcoded for now

        # modify c_skip, c_out to make F_x predict x0 for onehot variables
        # D_x predicts x0 - so if F_x is x0 then c_skip should be zero and c_out should be one
        pred_x0_lats = self.pred_x0.lats
        if pred_x0_lats.bool().any():
            pred_x0_lats = pred_x0_lats.to(x.device).to(torch.float32)
            c_skip = c_skip * (1 - pred_x0_lats)
            c_out = c_out * (1 - pred_x0_lats) + pred_x0_lats

        F_x = self.model((c_in * x).to(dtype), y, c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        # print('c_out', c_out[0, ::32**2*3])
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
