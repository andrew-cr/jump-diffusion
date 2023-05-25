import os
import torch
import torch.nn.functional as F
from .base import StructuredDatasetBase
from .image import ImageFolderDataset, base_image_dataset_kwargs, image_gettable_kwargs
import numpy as np
from torchvision.io import read_video
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import io
import h5py

from .base import StructuredDatasetBase
from .image import Dataset, base_image_dataset_kwargs, image_gettable_kwargs



class VideoDataset(Dataset):

    def __init__(self, path, num_frames, resolution=None, **kwargs):
        self._path = path
        self._num_frames = num_frames
        self._resolution = resolution
        self._video_fnames = self._find_videos(path)
        print('foudn fnames', self._video_fnames)

        # run super init
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._video_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None:
            assert raw_shape[-1] == resolution
        
        super().__init__(name=name, raw_shape=raw_shape, normalize=True, **kwargs)
        assert not any(self._xflip), "xflip not supported for video"

    def _find_videos(self, path):
        if os.path.isdir(path):
            self._all_fnames = {os.path.join(root, fname) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif path.endswith('.csv'):
            with open(path, 'r') as f:
                paths = [line.strip().split(',')[1] for line in f.readlines()]
                self._all_fnames = [os.path.join(os.path.dirname(path), os.path.basename(fname)) for fname in paths]
        else:
            raise ValueError(f"Cannot handle path of this type: {path}")
        return sorted(fname for fname in self._all_fnames if fname.endswith(self.ext))

    def _load_raw_image(self, raw_idx):
        path = self._video_fnames[raw_idx]
        loaded = self._load_video(path)
        if self._resolution is not None:
            loaded = F.interpolate(torch.from_numpy(loaded), size=self._resolution).numpy()
        return self._trim_num_frames(loaded)

    @property
    def image_shape(self):
        """
        VIDEO shape, not image shape
        """
        return list(self._raw_shape[1:])

    @property
    def num_frames(self):
        return self.image_shape[0]

    @property
    def num_channels(self):
        return self.image_shape[1]

    @property
    def resolution(self):
        assert len(self.image_shape) == 4
        assert self.image_shape[2] == self.image_shape[3]
        return self.image_shape[2]

    def _trim_num_frames(self, video):
        frames_found = video.shape[0]
        if self._num_frames is None:
            return video
        elif frames_found < self.num_frames:
            raise ValueError(f"Video has {frames_found} frames, but {self.num_frames} are required")
        else:
            raise ValueError("Better to specify task if you want num_frames < video length")

    def augment(self, data, augment_pipe):
        assert augment_pipe is None
        return data, None

video_dataset_kwargs = set.union(base_image_dataset_kwargs, set([('num_frames', 'int', None), ('task_distribution', 'str', None),]))
video_gettable_kwargs = image_gettable_kwargs


class FlexibleVideoDataset(VideoDataset, StructuredDatasetBase):

    def __init__(self, task_distribution=None, latent_space=True, duplicate_videos_in_batch=1, **kwargs):
        self.task_distribution = task_distribution
        self.duplicate_videos_in_batch = duplicate_videos_in_batch
        super().__init__(**kwargs)
        if latent_space:
            from .ldm_utils import load_vae
            image_resolution = self.resolution
            self.upsample = image_resolution < 256
            self.latent_resolution = 32 if self.upsample else self.resolution // 8
            self.vae = load_vae()
            self.latent_num_channels = self.vae.embed_dim
        self.dim_handler_class = VideoDimHandler

    def sample_some_indices(self, max_indices, T):
        s = torch.randint(low=1, high=max_indices+1, size=())
        max_scale = T / (s-0.999)
        scale = np.exp(np.random.rand() * np.log(max_scale))
        pos = torch.rand(()) * (T - scale*(s-1))
        indices = [int(pos+i*scale) for i in range(s)]
        # do some recursion if we have somehow failed to satisfy the consrtaints
        if all(i<T and i>=0 for i in indices):
            return indices
        else:
            print('warning: sampled invalid indices', [int(pos+i*scale) for i in range(s)], 'trying again')
            return self.sample_some_indices(max_indices, T)

    def _sample_task(self, video, deterministic=False):
        if self.task_distribution is None:
            # unconditional modelling of entire video
            indices = torch.arange(self.num_frames).float()
            return video, video[:0], indices
        pad_and_tensorize = lambda x, m: torch.tensor(x + [-1]*(m-len(x))).float()
        mode = self.task_distribution.split('-')[0]
        if mode == 'autoreg':
            _, n_obs, n_lat = self.task_distribution.split('-')
            start_index = np.random.randint(0, len(video)-int(n_lat)-int(n_obs)+1) if not deterministic else 0
            obs_indices = torch.arange(start_index, start_index + int(n_obs)).float()
            lat_indices = torch.arange(start_index + int(n_obs), start_index + int(n_lat) + int(n_obs)).float()
        elif mode == 'fdm':
            K = int(self.task_distribution.split('-')[1])
            lat_indices = []
            obs_indices = []
            while True:
                add_to_lat = (len(lat_indices) == 0) or (torch.rand(()) < 0.5)
                indices = self.sample_some_indices(max_indices=K, T=self.num_frames)
                indices = [i for i in indices if i not in lat_indices and i not in obs_indices]
                if len(indices) > K - len(obs_indices) - len(lat_indices):
                    break
                (lat_indices if add_to_lat else obs_indices).extend(indices)
            lat_indices = pad_and_tensorize(lat_indices, K)
            obs_indices = pad_and_tensorize(obs_indices, K)
        elif mode == 'even':
            K, d = (int(i) for i in self.task_distribution.split('-')[1:])  # max number of frames, max spacing between frames
            k = np.random.randint(2, K+1)
            offset = np.random.randint(0, self.num_frames - d*(k-1))
            lat_indices = list(range(offset, offset + k*d, d))
            lat_indices = pad_and_tensorize(lat_indices, K)
            obs_indices = pad_and_tensorize([], 1)  # dummy value - we never want observed stuff
        else:
            raise ValueError(f"Unknown task distribution {self.task_distribution}")
        lat = video[lat_indices.long()]
        obs = video[obs_indices.long()]
        indices = torch.cat([lat_indices, obs_indices])
        return lat, obs, indices

    def __getitem__(self, idx, will_augment=True, deterministic=False, do_duplicate=True):
        video, *etc = super().__getitem__(idx)
        return_vals = []
        for i in range(self.duplicate_videos_in_batch if do_duplicate else 1):
            if len(etc) > 0 and etc[0].numel() == 0:
                etc = etc[1:]  # remove empty labels
            lat, obs, indices = self._sample_task(video, deterministic=deterministic)
            if not will_augment and hasattr(self, 'vae'):
                lat, obs, indices = lat.cuda(), obs.cuda(), indices.cuda()
                lat = self.encode(lat)
                obs = self.encode(obs)
            return_vals.append((lat, obs, indices, *etc))
        return_val = tuple(torch.stack([rv[j] for rv in return_vals]) for j in range(len(return_vals[0]))) if do_duplicate else return_vals[0]
        return return_val

    def augment(self, data, augment_pipe):
        assert augment_pipe is None
        # encode to latent space
        lat, obs, *etc = data
        if hasattr(self, 'vae'):
            lat = self.encode(lat)
            obs = self.encode(obs)
        data = (lat, obs, *etc)
        return data, None

    def encode(self, t, decode=False):
        batch_dim = t.ndim == 5
        if not batch_dim:
            t = t.unsqueeze(0)
        B, T, C, H, W = t.shape
        t = t.view(-1, C, H, W)
        if not decode:
            if self.upsample:
                t = F.interpolate(t, size=(256, 256), mode='bilinear')
            t = self.vae.encode(t).sample()
            t = t/8  # NOTE scaling chosen slightly arbitrarily to make near-unit variance
        else:
            t = t*8  # NOTE scaling chosen slightly arbitrarily to make near-unit variance
            t = self.vae.decode(t)
        t = t.view(B, T, *t.shape[1:])
        if not batch_dim:
            t = t.squeeze(0)
        return t

    def decode(self, t):
        return self.encode(t, decode=True)

    def get_videos(self, data, mark_obs=False):
        lat, obs, indices, *etc = data
        print('getting videos, lats min max', lat.min(), lat.max())
        # decode lat and obs
        if hasattr(self, 'vae'):
            lat = self.decode(lat)
            obs = self.decode(obs)
        indices = indices.long()
        if mark_obs:  # add red border to obs
            obs = obs.clone()
            red = torch.tensor([1, -1, -1]).view(1, 1, 3, 1, 1)
            obs[:, :, :, :2, :] = red
            obs[:, :, :, -2:, :] = red
            obs[:, :, :, :, :2] = red
            obs[:, :, :, :, -2:] = red
        videos = torch.zeros_like(lat[:, :1]).repeat(1, indices.shape[1], 1, 1, 1)
        for b in range(videos.shape[0]):
            frames_b = torch.cat([lat[b], obs[b]], dim=0)
            indices_existing_b = sorted([i for i in indices[b] if i != -1])
            for frame, idx in zip(frames_b, indices[b]):
                if idx == -1:
                    continue
                local_idx = indices_existing_b.index(idx)
                videos[b, local_idx] = frame
        return videos

    def get_flat_videos(self, *args, **kwargs):
        videos = self.get_videos(*args, **kwargs)
        flat_videos = []
        for video in videos:
            # concatenate frames along width dimension
            T, C, H, W = video.shape
            video = video.permute(1, 2, 0, 3).contiguous().view(C, H, T*W)
            flat_videos.append(video)
        return torch.stack(flat_videos)


class VideoDimHandler():
    def __init__(self, shapes):
        self.lat_shape, self.obs_shape, self.indices_shape = shapes
        print('lat_shape', self.lat_shape, 'obs_shape', self.obs_shape, 'indices_shape', self.indices_shape)
        assert len(self.indices_shape) == 1
        print('shapes')
        print(shapes)
        self.max_dim = self.lat_shape[0]
        print('self.max_dim', self.max_dim)

    def count_dims(self, data):
        return self.get_dims(data).sum(dim=1)

    def get_dims(self, data):
        return (data[2] != -1).float()

    # def mask(self, dims=None, data=None, device=None):
    #     """
    #     Takes 1D tensor describing a set of dimensions. Returns a
    #     data-shaped thing with 1s for dimensions within that and 0s
    #     for dimensions outside of it.
    #     """
    #     print('woooOOOOOOoooOOooOOOOO')
    #     print('calling mask')
    #     print('woooOOOOOOoooOOooOOOOOoooooOOOOOOo')
    #     if dims is None:
    #         dims = self.get_dims(data)
    #     else:
    #         assert data is None
    #     indices = torch.tensor(dims, device=device)
    #     assert indices.ndim == 1, "dims should be a 1D tensor"
    #     n_lat = self.lat_shape[0]
    #     lat = torch.ones(self.lat_shape, device=device) * indices[:n_lat].view(-1, 1, 1, 1)
    #     obs = torch.ones(self.obs_shape, device=device) * indices[n_lat:].view(-1, 1, 1, 1)
    #     return lat, obs, indices

    def batched_mask(self, dimses=None, datas=None, device=None):
        """
        Takes a batch of 1D tensors describing a set of dimensions.
        Returns a data-shaped thing with 1s for dimensions within that
        and 0s for dimensions outside of it.
        """
        if dimses is None:
            B = datas[0].shape[0]
            dimses = torch.stack([self.get_dims(tuple(t[i] for t in datas)) for i in range(B)])
        else:
            dimses = torch.cat([dimses, torch.zeros_like(dimses[:, :1])], dim=1)
            B = dimses.shape[0]
            assert datas is None
        if device is None:
            device = dimses.device
        indiceses = dimses
        assert indiceses.ndim == 2, "dimses should be a 2D tensor"
        n_lat = self.lat_shape[0]
        lat = torch.ones((B, *self.lat_shape), device=device) * indiceses[:, :n_lat].view(B, -1, 1, 1, 1)
        obs = torch.ones((B, *self.obs_shape), device=device) * indiceses[:, n_lat:].view(B, -1, 1, 1, 1)
        return lat, obs, indiceses

    def which_dim_mask(self, device):
        """
        Returns a data-shaped thing with i in the ith dim for all i.
        Only necessarily makes sense for latent dimensions.
        """
        indices = torch.arange(self.max_dim+1, device=device, dtype=torch.long)
        n_lat = self.lat_shape[0]
        lat = torch.ones(self.lat_shape, device=device, dtype=torch.long) * indices[:n_lat].view(-1, 1, 1, 1)
        obs = torch.ones(self.obs_shape, device=device, dtype=torch.long) * indices[n_lat:].view(-1, 1, 1, 1)
        return lat, obs, indices


    def add_dim(self, data, add_index):
        """
        Given data and an index for adding a dimension, edits data to
        represent more dims (without setting particular values for the
        new dim). Returns edited data and a 1D tensor describing which dims
        are new.
        """
        lat, obs, indices, *etc = data
        n_lat = lat.shape[0]
        lat_indices = indices[:n_lat]
        assert -1 in lat_indices, "can't add a dim if all lat dims already exist"
        assert 0 in indices, "indices should contain a zero"
        assert (add_index-1) in indices, "can't add a dim without previous dim existing"
        add_pos = (indices == -1).nonzero().min()  # first non-existing dim
        indices[indices >= add_index] = indices[indices >= add_index] + 1
        indices[add_pos] = add_index
        data = (lat, obs, indices, *etc)
        added_dims = torch.zeros_like(indices[..., :self.max_dim])
        added_dims[add_pos] = 1
        return data, added_dims


class MinecraftVideoDataset(FlexibleVideoDataset):
    ext = "mp4"
    is_onehot = [0, 0, 0]
    is_image = [0, 0, 0]

    def _load_video(self, path):
        """
        return TxCxHxW tensor normed in [0, 255]
        """
        frames, audio, meta = read_video(str(path), pts_unit='sec')
        frames = torch.einsum('thwc->tchw', frames)
        return frames.numpy()

    def _load_raw_labels(self):
        raise NotImplementedError


class CarlaVideoDataset(FlexibleVideoDataset):
    ext = "pt"
    is_onehot = [0, 0, 0]
    is_image = [0, 0, 0]

    def _load_video(self, path):
        """
        return TxCxHxW tensor normed in [0, 255]
        """
        frames = torch.load(path)
        frames = torch.einsum('thwc->tchw', frames)
        return frames.numpy()

    def _load_raw_labels(self):
        raise NotImplementedError


class MazesVideoDataset(FlexibleVideoDataset):
    _res = 64
    ext = "pt"
    is_onehot = [0, 0, 0]
    is_image = [0, 0, 0]

    def _load_video(self, path):
        """
        return TxCxHxW tensor normed in [0, 255]
        """
        video = torch.load(path)
        byte_to_tensor = lambda x: ToTensor()(Resize(self._res)((Image.open(io.BytesIO(x)))))  # converts to range [0, 1]
        video = torch.stack([byte_to_tensor(frame) for frame in video])
        unnormed_video = video * 255
        return unnormed_video.numpy().astype(np.uint8)

    def _load_raw_labels(self):
        raise NotImplementedError


class Mazes32VideoDataset(MazesVideoDataset):
    _res = 32


class RoboDeskVideoDataset(FlexibleVideoDataset):
    is_onehot = [0, 0, 0]
    is_image = [0, 0, 0]

    def _find_videos(self, path):
        # find all hdf5 files in `path` directory
        all_files = sum([[os.path.join(root, f) for f in files]
                         for root, _, files in os.walk(path) for f in files], [])
        hdf5_files = [f for f in all_files if f.endswith('.hdf5') and 'noise_0.1' in f]
        # find list of videos within each hdf5 file
        paths = []
        for f in hdf5_files:
            with h5py.File(f, 'r') as hf:
                demo_keys = hf['data'].keys()
                for demo_key in demo_keys:
                    video = hf['data'][demo_key]['obs']['camera_image']
                    assert len(video.shape) == 4
                    paths.append('--'.join([f, 'data', demo_key, 'obs', 'camera_image']))
        return paths
    
    def _load_video(self, path):
        """
        return TxCxHxW tensor normed in [0, 255]
        """
        hdf5_file, *keys = path.split('--')
        with h5py.File(hdf5_file, 'r') as hf:
            layer = hf
            for key in keys:
                layer = layer[key]
            video = torch.tensor(np.array(layer))
        video = torch.einsum('thwc->tchw', video)
        return video.numpy()



flexible_video_dataset_kwargs = set.union(video_dataset_kwargs, set([('task_distribution', 'str', None), ('latent_space', 'bool', False), ('duplicate_videos_in_batch', 'int', 1),]))
# do not cache by default for video (bc videos are big...) -----
assert ('cache', 'bool', True) in flexible_video_dataset_kwargs
flexible_video_dataset_kwargs.remove(('cache', 'bool', True))
flexible_video_dataset_kwargs.add(('cache', 'bool', False))
# --------------------------------------------------------------
flexible_video_gettable_kwargs = video_gettable_kwargs


video_datasets_to_kwargs = {
    MinecraftVideoDataset: flexible_video_dataset_kwargs,
    CarlaVideoDataset: flexible_video_dataset_kwargs,
    MazesVideoDataset: flexible_video_dataset_kwargs,
    Mazes32VideoDataset: flexible_video_dataset_kwargs,
    RoboDeskVideoDataset: flexible_video_dataset_kwargs,
}
video_kwargs_gettable_from_dataset = {
    MinecraftVideoDataset: flexible_video_gettable_kwargs,
    CarlaVideoDataset: flexible_video_gettable_kwargs,
    MazesVideoDataset: flexible_video_gettable_kwargs,
    Mazes32VideoDataset: flexible_video_gettable_kwargs,
    RoboDeskVideoDataset: flexible_video_gettable_kwargs,
}
