import math
import numpy as np
import torch
import wandb

#----------------------------------------------------------------------------
#
# One sample from these datasets is data = (tensor1, tensor2, ...), where each tensor can have
# arbitrary shape. To feed them into e.g. the diffusion sampler, a batch of them can be flattened 
# by Structure.flatten_batch, which returns (lats, obs), where lats is a B x latent_dim tensor of
# all non-conditioned on parts of the data and obs is a tuple of (arbitrarily-shaped) tensors of
#  all parts of the data that are conditioned on. This operation is undone with 
# Structure.unflatten_batch(lats, obs). Throughout this codebase, lats is often referred to as
# x and obs as y.


class StructuredDatasetBase():
    is_onehot = None  # must be set by subclass

    def __getitem__(self, will_augment):
        raise NotImplementedError

    def _unnormalise_images(self, images):
        return (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def get_images(self, data):
        raise NotImplementedError

    def log_batch(self, data, return_dict=False):
        d = {}
        for i, tensor in enumerate(data):
            if tensor is None:
                continue  # marginalised
            if tensor.isnan().any():
                print('not loggin nan tensor')
                continue
            if self.is_image[i]:
                d[f"Samples/images_{i}"] = wandb.Image(gridify_images(self._unnormalise_images(tensor)))
            elif self.is_onehot[i]:
                d[f"Samples/onehot_{i}_raw"] = wandb.Histogram(tensor.cpu().numpy())
                d[f"Samples/onehot_{i}"] = wandb.Histogram(torch.argmax(tensor, dim=1).cpu().numpy())
            else:
                d[f"Samples/tensor_{i}"] = wandb.Histogram(tensor.cpu().numpy())
        if return_dict:
            return d
        wandb.log(d)

    def load_network_state_dict(*args, **kwargs):
        # Assume no state
        pass

class GraphicalStructureBase():

    def adjust_st_batch(self, st_batch):
        # for things like setting CoM=0 for molecules
        pass

    def get_auto_target(self, st_batch, adjust_val):
        return st_batch.get_flat_lats()

    def get_auto_target_IS(self, xt_dp1_st_batch, adjust_val_dp1, adjust_val_d):
        return xt_dp1_st_batch.get_flat_lats()



def gridify_images(images):
    B_ = images.shape[0]
    rows = math.ceil(math.sqrt(B_))
    cols = math.ceil(B_ / rows)
    images = np.concatenate([images, np.zeros([rows*cols - B_, *images.shape[1:]])], axis=0)
    B, H, W, C = images.shape
    images = images.reshape(rows, cols, H, W, C)  # rows cols H W C
    # reshape to rows*H x cols*W x C
    images = np.concatenate([
        np.concatenate([images[r, c] for c in range(cols)], axis=1)
        for r in range(rows)
    ], axis=0)
    return images
