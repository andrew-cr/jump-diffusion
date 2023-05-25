import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from training.structure import StructuredDataBatch
from training.diffusion_utils import get_rate_using_x0_pred

def get_timestep_embedding(timesteps, embedding_dim, max_timesteps=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(max_timesteps) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, channels,
                 dropout, temb_channels=512, node_emb_channels=None):
        super().__init__()
        make_linear = lambda a, b: nn.Conv1d(a, b, kernel_size=1, stride=1, padding=0)
        self.channels = channels
        self.norm1 = Normalize(channels)
        # rewrite the below using make_linear
        self.conv1 = make_linear(channels, channels)
        self.temb_proj = make_linear(temb_channels, channels)
        if node_emb_channels is not None:
            self.var_proj = make_linear(node_emb_channels, channels)
        self.norm2 = Normalize(channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_linear(channels, channels)

    def forward(self, x, temb, node_emb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))
        if node_emb is not None:
            h = h + self.var_proj(node_emb)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=1, attn_dim_reduce=1):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels//attn_dim_reduce,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def sparse_forward(self, x, sparse_attention_mask_and_indices):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, n = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, n)
        # beta = (int(c//heads)**(-0.5)) # standard attention scaling
        # unnormalized attention
        beta = 1
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)

        valid_indices_mask, attendable_indices = sparse_attention_mask_and_indices
        b_mask, nq, max_attendable_keys = valid_indices_mask.shape
        assert b_mask == b or b_mask == 1
        attendable_indices = attendable_indices.view(1, 1, nq, max_attendable_keys).expand(b, heads, nq, max_attendable_keys)
        def get_keys_or_values(t, indices):
            *batch_shape, nd, nv = t.shape
            t = t.transpose(-1, -2)\
                .view(*batch_shape, nv, 1, nd)\
                .expand(*batch_shape, nv, max_attendable_keys, nd)
            index = indices.view(*batch_shape, nv, max_attendable_keys, 1)\
                .expand(-1, -1, -1, -1, c//heads)
            return t.gather(dim=2, index=index)

        attended_keys = get_keys_or_values(k, indices=attendable_indices)   # b x heads x h*w x max_attendable_keys x c
        attended_values = get_keys_or_values(v, indices=attendable_indices)

        weights = beta * torch.einsum('bhqkc,bhcq->bhqk', attended_keys, q)
        inf_matrix = torch.zeros_like(valid_indices_mask)
        inf_matrix[valid_indices_mask==0] = torch.inf
        weights = weights - inf_matrix.view(b_mask, 1, nq, max_attendable_keys)
        weights = weights.softmax(dim=-1)

        h_ = torch.einsum('bhqk,bhqkc->bhqc', weights, attended_values)
        h_ = h_.permute(0, 3, 1, 2).reshape(b, c, n)
        h_ = self.proj_out(h_)
        out = x+h_
        return out, None

    def forward(self, x, sparsity_matrix=None, sparse_attention_mask_and_indices=None, return_w=False):
        if sparse_attention_mask_and_indices is not None:
            out, w_ = self.sparse_forward(x, sparse_attention_mask_and_indices)
            return (out, w_) if return_w else out
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, n = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, n)
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)
        w_ = torch.einsum('bhdk,bhdq->bhqk', k, q)
        w_ = w_ * (int(c//heads)**(-0.5))
        if sparsity_matrix is not None:
            inf_matrix = torch.zeros_like(sparsity_matrix)
            inf_matrix[sparsity_matrix==0] = torch.inf
            w_ = w_ - inf_matrix.view(-1, 1, n, n)
        w_ = torch.nn.functional.softmax(w_, dim=3)
        h_ = torch.einsum('bhdk,bhqk->bhdq', v, w_)
        h_ = h_.view(b, c, n)
        h_ = self.proj_out(h_)
        out = x+h_
        return (out, w_) if return_w else out

