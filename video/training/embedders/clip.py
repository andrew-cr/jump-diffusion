import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


clip_names_to_kwargs = {}


class ClipEmbedder(nn.Module):       
    def __init__(self, project_to_sphere=True, stddev=1, proj_clip_to=-1):
        super().__init__()
        clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
        self.image_embedder = clip_model.visual
        self.norm = preprocess.transforms[-1]
        self.project_to_sphere = project_to_sphere
        self.stddev = stddev
        self.proj = nn.Linear(512, proj_clip_to) if proj_clip_to > 0 else nn.Identity()

    def forward(self, data):
        images, *_ = data
        inp = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        inp = self.norm((inp+1)/2) # initially normed to [-1, 1]. Change to [0, 1] and then use self.norm
        emb = self.proj(self.image_embedder(inp))
        if self.project_to_sphere:
            emb = self.stddev * F.normalize(emb, dim=-1) * torch.sqrt(torch.tensor(emb.shape[-1]))
        data = (*data, emb)
        return data

    def set_requires_grad(self, val):
        for param in self.parameters():
            param.requires_grad = val 

clip_embedder_kwargs = set([('project_to_sphere', 'bool', True),
                            ('stddev', 'float', 0.5),
                            ('proj_clip_to', 'int', -1),])
clip_names_to_kwargs['ClipEmbedder'] = clip_embedder_kwargs


class ClipChannels(ClipEmbedder):

    def forward(self, data):
        images, *etc, clip_emb = super().forward(data)
        B, C = clip_emb.shape
        _, _, H, W = images.shape
        clip_emb = clip_emb.reshape(B, C, 1, 1).expand(B, C, H, W)
        # SNR for clip_emb now scaled by sqrt(H*W)
        return (images, *etc, clip_emb)

clip_names_to_kwargs['ClipChannels'] = clip_embedder_kwargs
