import os
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    """ Copied from the Stabel diffusion repo.
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_vae():
    """ Loads the Stable diffusion VAE.
    """
    stable_diffusion_path = os.environ["STABLE_DIFFUSION_PATH"]
    config_path = os.path.join(stable_diffusion_path, "configs/stable-diffusion/v1-inference.yaml")
    ckpt_path = os.path.join(stable_diffusion_path, "models/ldm/stable-diffusion-v1/sd-v1-1-full-ema.ckpt")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    for param in model.parameters():
        param.requires_grad = False
    return model.first_stage_model