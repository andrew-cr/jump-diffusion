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

    def __getitem__(self, will_augment, deterministic=False):
        raise NotImplementedError

    def _unnormalise_images(self, images, is_video=False):
        images = images.permute(0, 1, 3, 4, 2) if is_video else images.permute(0, 2, 3, 1)
        return (images * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()

    def get_images(self, data):
        raise NotImplementedError

    def log_batch(self, data, emb_data=None, return_dict=False):
        d = {}
        if hasattr(self, 'get_videos'):  #self.is_video[i]:
            vid = gridify_images(self._unnormalise_images(self.get_videos(data, mark_obs=True), is_video=True))
            d[f"Samples/video"] = wandb.Video(vid, fps=10, format="mp4")
        for i, tensor in enumerate(data):
            if tensor is None:
                continue  # marginalised
            if self.is_image[i]:
                d[f"Samples/images_{i}"] = wandb.Image(gridify_images(self._unnormalise_images(tensor)))
            elif self.is_onehot[i]:
                d[f"Samples/onehot_{i}_raw"] = wandb.Histogram(tensor.cpu().numpy())
                d[f"Samples/onehot_{i}"] = wandb.Histogram(torch.argmax(tensor, dim=1).cpu().numpy())
            elif hasattr(self, 'is_embedding') and self.is_embedding[i]:
                columns = [f"embedding_{i}_{j}" for j in range(tensor.shape[1])]
                for j in range(min(10, tensor.shape[1])):
                    d[f"Samples/embedding_{i}_{j}"] = wandb.Histogram(tensor[:, j].cpu().numpy())
            else:
                d[f"Samples/tensor_{i}"] = wandb.Histogram(tensor.cpu().numpy())
        if emb_data is not None:
            for i, tensor in enumerate(emb_data):
                if tensor is not None:
                    d[f"Samples/emb_{i}"] = wandb.Histogram(tensor.cpu().numpy())
        if return_dict:
            return d
        wandb.log(d)


def gridify_images(images):  # or videos
    is_video = len(images.shape) == 5
    B_ = images.shape[0]
    rows = math.ceil(math.sqrt(B_))
    cols = math.ceil(B_ / rows)
    images = np.concatenate([images, np.zeros([rows*cols - B_, *images.shape[1:]])], axis=0)
    if not is_video:
        images = images[:, np.newaxis]  # add time dimension
    B, T, H, W, C = images.shape
    images = images.reshape(rows, cols, T, H, W, C)
    # reshape to rows*H x cols*W x C
    images = np.concatenate([
        np.concatenate([images[r, c] for c in range(cols)], axis=2)
        for r in range(rows)
    ], axis=1)
    if is_video:
        images = images.transpose(0, 3, 1, 2)  # channel is expected before spatial dims
    else:
        images = images.squeeze(0)  # remove time dimension
    return images
