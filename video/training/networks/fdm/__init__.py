import torchvision.models.video.resnet as rn
from einops.layers.torch import Rearrange
from einops import rearrange
from training.networks.fdm.unet import *
from torch_utils import persistence
from PIL import Image
import os


fdm_name_to_kwargs = {}
fdm_name_to_kwargs['fdm'] = set([
    ('model_type', 'str', 'WrappedUNetVideoModel'), ('model_channels', 'int', 128),
    ('num_res_blocks', 'int', 1), ('attention_resolutions', 'parse_int_list', (16, 8)),
    ('conv_resample', 'bool', True), ('use_checkpoint', 'bool', False),
    ('num_heads', 'int', 4), ('num_heads_upsample', 'int', -1),
    ('use_scale_shift_norm', 'bool', True), ('use_rpe_net', 'bool', True),
    ('detach_jump_grads', 'bool', True), ('only_use_neighbours', 'int', 0),
])
fdm_name_to_kwargs['just_jump'] = set([
    ('model_type', 'str', 'VideoJumpPredictor'), ('jump_net_embedder_type', 'str', 'unet'),
])


"""
Assume we get from the dataset:

- latent frames
- observed frames
- all frame indices (in which -1 indicates padding frames)
- (optionally) additional conditioning info. for all frames

"""


@persistence.persistent_class
class WrappedUNetVideoModel(UNetVideoModel):
    def __init__(self, structure, augment_dim=0, only_use_neighbours=0, **kwargs):
        print("WrappedUNetVideoModel")
        print(kwargs)
        self.structure = structure
        assert not self.structure.observed[0]
        assert self.structure.observed[1]
        assert self.structure.observed[2]
        assert augment_dim == 0
        data_channels = structure.shapes[0][1]
        data_width = structure.shapes[0][2]
        if data_width == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif data_width == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif data_width == 64:
            channel_mult = (1, 2, 3, 4)
        elif data_width == 32:
            channel_mult = (1, 2, 2, 2)
        self.jump_diffusion = kwargs['jump_diffusion']
        if self.jump_diffusion:
            kwargs['jump_diffusion_max_dim'] = self.structure.dim_handler.max_dim
        self.only_use_neighbours = only_use_neighbours
        super().__init__(in_channels=data_channels, out_channels=data_channels, image_size=data_width, dropout=0.0,
                         channel_mult=channel_mult, **kwargs)

    def forward(self, x, y, noise_labels, augment_labels=None):
        if augment_labels is not None:
            raise NotImplementedError
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        latent_frames, observed_frames, frame_indices, *etc, dummy_x0_dims, dummy_dels_between = data
        B, T, C, H, W = latent_frames.shape
        unshuffled_noise_labels = noise_labels.view(B, T)
        if len(etc) != 0:
            raise NotImplementedError
        if self.only_use_neighbours:
            # delete from frame_indices so that the only ones remaining are those next to a frame with noise (many don't have noise if using highly_nonisotropic)
            new_frame_indices = -1 * th.ones_like(frame_indices)[:,:-1]
            for b in range(frame_indices.shape[0]):
                assert frame_indices[b][-1] == -1, "This lets us do line below"
                latent_frame_indices_b = frame_indices[b][:-1]
                existing_indices = frame_indices[b][frame_indices[b] != -1]
                # print('unshuffled_noise_labels', unshuffled_noise_labels.shape)
                noise_vals = unshuffled_noise_labels[b][latent_frame_indices_b != -1].unique()  # get unique values of noise_labels that are not -1
                if len(noise_vals) == 1:
                    # only one frame type - it must have non-zero noise
                    #nprint('only one noise val', noise_vals[0])
                    new_frame_indices = frame_indices[:,:-1]
                    continue
                elif len(noise_vals) == 2:
                    # print('two noise vals', noise_vals)
                    zero_noise_val = noise_vals.min()  # value of noise_label corresponding to sigma=0
                    nonzero_noise_mask = (unshuffled_noise_labels[b] != zero_noise_val).float() * (latent_frame_indices_b != -1).float()
                    nonzero_noise_indices = latent_frame_indices_b[nonzero_noise_mask.bool()]
                    for index in nonzero_noise_indices:
                        # keep this index obviously
                        new_frame_indices[b, latent_frame_indices_b==index] = index
                        # keep nearest neighbour before
                        all_before = existing_indices[existing_indices < index]
                        if len(all_before) != 0:
                            k = min(self.only_use_neighbours, len(all_before))
                            nearest_befores = all_before.topk(k, largest=True).values
                            for nearest_before in nearest_befores:
                                new_frame_indices[b, latent_frame_indices_b==nearest_before] = nearest_before
                        # keep nearest neighbour after
                        all_after = existing_indices[existing_indices > index]
                        if len(all_after) != 0:
                            k = min(self.only_use_neighbours, len(all_after))
                            nearest_afters = all_after.topk(k, largest=False).values
                            for nearest_after in nearest_afters:
                                new_frame_indices[b, latent_frame_indices_b==nearest_after] = nearest_after
                    #nprint('original row:', frame_indices[b])
                    #nprint('made row of frame indices', new_frame_indices[b])
                    #nprint()
                else:
                    raise Exception("Shouldn't happen")
            frame_indices = th.cat([new_frame_indices, th.ones_like(frame_indices[:, -1:]) * -1], dim=1)  # add back on the (always non-existent) index for obs frame
        if self.jump_diffusion:
            # change the indices to be consecutive --------------------
            frame_indices = frame_indices.clone()
            for b in range(frame_indices.shape[0]):
                existing_indices = frame_indices[b, frame_indices[b] != -1]
                relative_indices = [sum(existing_indices < i) for i in existing_indices]
                frame_indices[b, frame_indices[b] != -1] = th.tensor(relative_indices).to(frame_indices.device).float()
            # ---------------------------------------------------------
        max_latent = latent_frames.shape[1]
        max_obs = observed_frames.shape[1]
        latent_indices = frame_indices[:, :max_latent]
        observed_indices = frame_indices[:, max_latent:max_latent+max_obs]
        max_exist = ((latent_indices != -1).sum(dim=1) + (observed_indices != -1).sum(dim=1)).max().item()
        frames = th.zeros((B, max_exist, C, H, W), device=x.device)
        indices = -1 * th.ones_like(frame_indices[:, :max_exist])
        obs_mask = th.zeros_like(indices).view(B, max_exist, 1, 1, 1)
        latent_mask = th.zeros_like(indices).view(B, max_exist, 1, 1, 1)
        noise_labels = th.zeros_like(noise_labels).view(B, T)
        for b in range(B):
            latents_exist = (latent_indices[b] != -1)
            n_lat = latents_exist.sum().item()
            frames[b, :n_lat] = latent_frames[b, latents_exist]
            indices[b, :n_lat] = latent_indices[b, latents_exist]
            noise_labels[b, :n_lat] = unshuffled_noise_labels[b, latents_exist]
            latent_mask[b, :n_lat] = 1
            observed_exist = (observed_indices[b] != -1)
            n_obs = observed_exist.sum().item()
            frames[b, n_lat:n_lat+n_obs] = observed_frames[b, observed_exist]
            indices[b, n_lat:n_lat+n_obs] = observed_indices[b, observed_exist]
            obs_mask[b, n_lat:n_lat+n_obs] = 1
        # -----------------------------
        # visualize the network input
        T = frames.shape[1]
        # vis_path = f'jump-net-input-{T}-training-{self.training}.png'
        if hasattr(self, 'plot_path') and not os.path.exists(self.plot_path):
            assert augment_labels is None
            with th.no_grad():
                vis_frames = rearrange(frames, 'b t c h w -> (b h) (t w) c')
                vis_frames = vis_frames.clip(-1, 1) * 127.5 + 127.5
                Image.fromarray(vis_frames.cpu().numpy().astype(np.uint8)).save(self.plot_path)
            # print()
            # print('SAVING TO ', self.plot_path)
            # print('indices')
            # print(indices)
            # print('noise_labels')
            # print(noise_labels.view(B, -1))
            # print()
        # -----------------------------
        outputs = super().forward(x=frames, x0=frames, timesteps=noise_labels, frame_indices=indices,
                                  obs_mask=obs_mask, latent_mask=latent_mask)
        output = outputs[0]
        if self.jump_diffusion and not self.only_use_neighbours:
            insertion_probs, x0_dim_pred = outputs[2:]
        elif self.only_use_neighbours:
            # we get nans if we try to do the jump stuff too, so let's not bother
            x0_dim_pred = dummy_x0_dims
            insertion_probs = dummy_dels_between

        # put the output back into the right place
        reconstructed_latents = th.zeros_like(latent_frames)
        for b in range(B):
            latents_exist = (latent_indices[b] != -1)
            n_lat = latents_exist.sum().item()
            reconstructed_latents[b, latents_exist] = output[b, :n_lat]
        data = [reconstructed_latents, None, None]
        if self.jump_diffusion:
            data = (*data, x0_dim_pred, insertion_probs)
        out = self.structure.flatten_latents(data, contains_marg=False)
        return out

    def set_requires_grad(self, val, freeze_pretrained=False):
        for name, param in self.named_parameters():
            param.requires_grad = val


def replace_bn_with_groupnorm(module, num_groups=32):
    """
    Recursively replaces all instances of `nn.BatchNorm3d` with `nn.GroupNorm`.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            num_channels = child.num_features
            while num_channels % num_groups != 0:
                num_groups //= 2
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            replace_bn_with_groupnorm(child, num_groups=num_groups)


class Conv3DNoTemporalStride(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


@persistence.persistent_class
class VideoJumpPredictor(nn.Module):
    def __init__(self, structure, augment_dim=0, jump_net_embedder_type='unet', **kwargs):
        print("WrappedUNetVideoModel")
        print(kwargs)
        self.structure = structure
        assert not self.structure.observed[0]
        assert self.structure.observed[1]
        assert self.structure.observed[2]
        assert augment_dim == 0
        data_channels = structure.shapes[0][1]
        data_width = structure.shapes[0][2]
        self.jump_diffusion = kwargs['jump_diffusion']

        # network should be like VGG but with 3D convolutions
        super().__init__()
        self.jump_net_embedder_type = jump_net_embedder_type
        if jump_net_embedder_type == 'resnet':
            self.video_embedder = rn._video_resnet(rn.BasicBlock, [Conv3DNoTemporalStride,]*4, [2, 2, 2, 2], rn.BasicStem, None, False)
            replace_bn_with_groupnorm(self.video_embedder)
            del self.video_embedder.fc
            del self.video_embedder.avgpool
            emb_res = data_width // 16
            flat_channels = 512 * emb_res**2
        elif jump_net_embedder_type == 'unet':
            self.video_embedder = UNetVideoModel(
                in_channels=3,
                model_channels=32,
                out_channels=3, 
                num_res_blocks=1, 
                attention_resolutions=[16, 8],
                conv_resample=True,
                use_checkpoint=False,
                num_heads=4,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                use_rpe_net=True,
                detach_jump_grads=False,  
                return_middle=True,
                channel_mult=(1, 4, 8, 16),
            )
            emb_res = data_width // self.video_embedder.middle_ds
            flat_channels = 16*32 * emb_res**2
        max_dims = structure.dim_handler.max_dim
        self.convs = nn.Sequential(
            Rearrange('b c t h w -> b (c h w) t'),
            nn.Conv1d(flat_channels, 128, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        self.dim_predictor = nn.Sequential(
            nn.Conv1d(128, max_dims+1, 3, padding=1),  # extra channel because index predictor
        )
        self.x0_dim_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b c 1 -> b c'),
            nn.Linear(128, max_dims),
        )

    def run_embedder(self, x, noise_labels, **kwargs):
        em = self.video_embedder
        if self.jump_net_embedder_type == 'resnet':
            x = rearrange(x, 'b t c h w -> b c t h w')
            return em.layer4(em.layer3(em.layer2(em.layer1(em.stem(x)))))
        elif self.jump_net_embedder_type == 'unet':
            out = em(x, x0=x, timesteps=noise_labels, **kwargs)
            return rearrange(out, 'b t c h w -> b c t h w')
        raise NotImplementedError

    def forward(self, x, y, noise_labels, augment_labels=None):
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        latent_frames, observed_frames, frame_indices, *etc, dummy_x0_dims, dummy_dels_between = data
        if len(etc) != 0:
            raise NotImplementedError
        if self.jump_diffusion:
            # change the indices to be consecutive --------------------
            frame_indices = frame_indices.clone()
            for b in range(frame_indices.shape[0]):
                existing_indices = frame_indices[b, frame_indices[b] != -1]
                relative_indices = [sum(existing_indices < i) for i in existing_indices]
                frame_indices[b, frame_indices[b] != -1] = th.tensor(relative_indices).to(frame_indices.device).float()
            # ---------------------------------------------------------
        max_latent = latent_frames.shape[1]
        max_obs = observed_frames.shape[1]
        latent_indices = frame_indices[:, :max_latent]
        observed_indices = frame_indices[:, max_latent:max_latent+max_obs]
        max_exist = ((latent_indices != -1).sum(dim=1) + (observed_indices != -1).sum(dim=1)).max().item()
        B, T, C, H, W = latent_frames.shape
        frames = th.zeros((B, max_exist, C, H, W), device=x.device)
        indices = -1 * th.ones_like(frame_indices[:, :max_exist])
        obs_mask = th.zeros_like(indices).view(B, max_exist, 1, 1, 1)
        latent_mask = th.zeros_like(indices).view(B, max_exist, 1, 1, 1)
        unshuffled_noise_labels = noise_labels.view(B, T)
        noise_labels = th.zeros_like(noise_labels).view(B, T)
        for b in range(B):
            latents_exist = (latent_indices[b] != -1)
            n_lat = latents_exist.sum().item()
            frames[b, :n_lat] = latent_frames[b, latents_exist]
            indices[b, :n_lat] = latent_indices[b, latents_exist]
            noise_labels[b, :n_lat] = unshuffled_noise_labels[b, latents_exist]
            latent_mask[b, :n_lat] = 1
            observed_exist = (observed_indices[b] != -1)
            n_obs = observed_exist.sum().item()
            frames[b, n_lat:n_lat+n_obs] = observed_frames[b, observed_exist]
            indices[b, n_lat:n_lat+n_obs] = observed_indices[b, observed_exist]
            obs_mask[b, n_lat:n_lat+n_obs] = 1
        # -----------------------------
        # visualize the network input
        T = frames.shape[1]
        # vis_path = f'jump-net-input-{T}-training-{self.training}.png'
        if hasattr(self, 'plot_path') and not os.path.exists(self.plot_path):
            assert augment_labels is None
            with th.no_grad():
                vis_frames = rearrange(frames, 'b t c h w -> (b h) (t w) c')
                vis_frames = vis_frames.clip(-1, 1) * 127.5 + 127.5
                Image.fromarray(vis_frames.cpu().numpy().astype(np.uint8)).save(self.plot_path)
            print()
            print('SAVING TO ', self.plot_path)
            # print('indices')
            # print(indices)
            # print('noise_labels')
            # print(noise_labels.view(B, -1))
            # print()
        # -----------------------------
        # we just operate on frames, ignore noise_labels + indices + masks
        score_pred = th.zeros_like(frames)
        emb = self.run_embedder(frames, noise_labels, frame_indices=indices, obs_mask=obs_mask, latent_mask=latent_mask)

        # reorder emb according to indices (so that later convolutions make sense)
        ordered_emb = th.zeros_like(emb)
        for b in range(B):
            n_lat = (indices[b] != -1).sum().item()
            indices_b = indices[b, :n_lat].long()
            for i, index in enumerate(indices_b):
                #print('moving index', i, 'to', index, 'in batch', b, 'of', B)
                ordered_emb[b, index] = emb[b, i]
        emb = ordered_emb
        
        emb = self.convs(emb)
        _x0_dim_pred = self.dim_predictor(emb)  # 
        x0_dim_pred = _x0_dim_pred[:, :-1, :]  # B x T x (Tx1)
        index_pred = _x0_dim_pred[:, -1:, :]  # last layer forms our index prediction

        for b in range(B):
            latents_exist = (latent_indices[b] != -1)
            n_lat = latents_exist.sum().item()
            max_number_of_dims_to_add = self.structure.dim_handler.max_dim - n_lat
            x0_dim_pred[b, max_number_of_dims_to_add+1:] = -1e6  # cannot exceed max dims allowed by in dataset
            # index_pred[b, :, latent_indices[b]==-1] = -1e6  # cannot add in position that doesn't exist

        B, T, C, H, W = frames.shape
        x0_dim_pred = rearrange(x0_dim_pred, 'b max_t t_plus_1 -> b t_plus_1 max_t')
        B, T_plus_1, max_T = x0_dim_pred.shape
        assert T <= max_T
        assert T_plus_1 == T + 1
        index_pred = rearrange(index_pred, 'b 1 t_plus_1 -> b t_plus_1 1')

        combined_index_pred = th.cat([x0_dim_pred, index_pred], dim=2)  # B x (T+1) x (max_T+1)
        n_to_pad = self.structure.dim_handler.max_dim + 1 - T_plus_1
        padding = th.zeros_like(combined_index_pred[:, :1], device=x.device).expand(B, n_to_pad, max_T+1).to(x.device)
        combined_index_pred = th.cat([combined_index_pred, padding], dim=1)

        actual_x0_dim_pred = self.x0_dim_predictor(emb)  # B x T

        # copied from WrappedUNetVideoModel - just formatting the output
        reconstructed_latents = th.zeros_like(latent_frames)
        for b in range(B):
            latents_exist = (latent_indices[b] != -1)
            n_lat = latents_exist.sum().item()
            reconstructed_latents[b, latents_exist] = score_pred[b, :n_lat]
        data = [reconstructed_latents, None, None]
        if self.jump_diffusion:
            data = (*data, actual_x0_dim_pred, combined_index_pred)
        out = self.structure.flatten_latents(data, contains_marg=False)
        return out

    def set_requires_grad(self, val, freeze_pretrained=False):
        for name, param in self.named_parameters():
            param.requires_grad = val
