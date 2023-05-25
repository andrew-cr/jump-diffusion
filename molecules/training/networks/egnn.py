import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from training.dataset.qm9 import get_cfg, get_dataset_info
from torch.distributions.categorical import Categorical
from training.egnn_utils import assert_mean_zero_with_mask, check_mask_correct, EGNN_dynamics_QM9, Jump_EGNN_QM9
from training.networks.gsdm import AttnBlock, ResnetBlock, get_timestep_embedding
from training.diffusion_utils import get_rate_using_x0_pred

args = get_cfg()
dataset_info = get_dataset_info(args.dataset, args.remove_h)

class EGNNMultiHeadJump(nn.Module):
    """
        EGNN backbone that gives score then a second network on the top that gives
        the rate and nearest atom prediction and a vector 

        detach_last_layer: whether to stop grad between EGNN and head net
    """
    def __init__(self, structure, detach_last_layer, rate_use_x0_pred,
                 n_attn_blocks, n_heads, transformer_dim,
                 noise_embed='ts', augment_dim=-1):
        super().__init__()
        self.structure = structure
        self.detach_last_layer = detach_last_layer
        

        args.context_node_nf = 0
        in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
        # in_node_nf is for atom types and atom charges
        # +1 for time
        dynamics_in_node_nf = in_node_nf + 1

        self.egnn_net = Jump_EGNN_QM9(
            in_node_nf=dynamics_in_node_nf, context_node_nf=6,
            n_dims=3, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
            CoM0=True, return_last_layer=True
        )

        self.rate_use_x0_pred = rate_use_x0_pred
        if self.rate_use_x0_pred:
            self.rdim = self.structure.graphical_structure.max_problem_dim
        else:
            self.rdim = 1

        self.transformer_dim = transformer_dim
        self.temb_dim = self.transformer_dim

        self.temb_net = nn.Linear(self.temb_dim, self.temb_dim)

        self.transformer_1_proj_in = nn.Linear(
            self.egnn_net.egnn.hidden_nf + 6, self.transformer_dim
        )

        # these are for the head that does the rate and nearest atom prediction
        self.attn_blocks = nn.ModuleList([
            AttnBlock(self.transformer_dim, n_heads, attn_dim_reduce=1)
            for _ in range(n_attn_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.transformer_dim,
                             dropout=0, temb_channels=self.temb_dim)
            for _ in range(n_attn_blocks)
        ])

        self.pre_rate_proj = nn.Linear(self.transformer_dim, self.transformer_dim)
        self.post_rate_proj = nn.Linear(self.transformer_dim, self.rdim)

        self.near_atom_proj = nn.Linear(self.transformer_dim, 1)

        # this is for the head that gives the vector given the nearest atom and std
        self.vec_transformer_in_proj = nn.Linear(
            self.egnn_net.egnn.hidden_nf + 6 + 1 + 2, self.transformer_dim
        )
        self.vec_attn_blocks = nn.ModuleList([
            AttnBlock(self.transformer_dim, n_heads, attn_dim_reduce=1)
            for _ in range(n_attn_blocks)
        ])

        self.vec_res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.transformer_dim,
                             dropout=0, temb_channels=self.temb_dim)
            for _ in range(n_attn_blocks)
        ])
        self.vec_weighting_proj = nn.Linear(self.transformer_dim, 1)

        self.pre_auto_proj = nn.Linear(self.transformer_dim, self.transformer_dim)
        self.post_auto_proj = nn.Linear(self.transformer_dim, 2*5 + 2 + 1)



    def forward(self, st_batch, ts, nearest_atom, sample_nearest_atom=False, augment_labels=None, forward_rate=None, rnd=None):
        # if sample_nearest_atom is true then we sample the nearest atom from the predicted distribution
        # and use that for the second head network. Use this during sampling but not during training


        # ts can pass directly as (B,) and should be normalized to [0,1]
        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2) # (B, n_nodes, n_features)
        x = st_batch.tuple_batch[0]
        dims = st_batch.get_dims()
        device = st_batch.get_device()
        B, n_nodes, _ = x.shape


        assert x.shape == (B, n_nodes, 3)

        atom_mask = torch.arange(st_batch.gs.max_problem_dim).view(1, -1) < dims.view(-1, 1) # (B, n_nodes)
        atom_mask = atom_mask.to(device)

        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) # (B, n_nodes_aug, n_nodes_aug) is 1 when both col and row are 1
        assert edge_mask.shape == (B, n_nodes, n_nodes)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=device).unsqueeze(0)
        assert diag_mask.shape == (1, n_nodes, n_nodes)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(B * n_nodes * n_nodes, 1)

        atom_mask = atom_mask.long().to(device)
        edge_mask = edge_mask.long().to(device)

        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)
        atom_type_one_hot = st_batch.tuple_batch[1]
        assert atom_type_one_hot.shape == (B, n_nodes, 5)
        charges = st_batch.tuple_batch[2]
        assert charges.shape == (B, n_nodes)
        charges = charges.view(B, n_nodes, 1)

        context_parts = torch.cat([
            *(st_batch.tuple_batch[i] for i in range(3, len(st_batch.tuple_batch)))
        ], dim=1)
        assert context_parts.shape == (B, 6)
        context_parts = context_parts.view(B, 1, 6).repeat(1, n_nodes, 1) # (B, n_nodes, 6)
        context_parts = context_parts * node_mask

        assert_mean_zero_with_mask(x, node_mask)
        check_mask_correct([x, atom_type_one_hot, charges, context_parts], node_mask)

        # note the time gets added on by Jump_EGNN_QM9
        xh = torch.cat([x, atom_type_one_hot, charges], dim=2)
        assert xh.shape == (B, n_nodes, 3+5+1)

        net_out, net_last_layer = self.egnn_net(
            t=ts, xh=xh, node_mask=node_mask, edge_mask=edge_mask, context=context_parts
        )
        
        assert net_out.shape == (B, n_nodes, 3+5+1)
        x_out = net_out[:, :, 0:3]
        atom_type_one_hot_out = net_out[:, :, 3:8]
        charges_out = net_out[:, :, 8:9]

        D_xt = torch.cat([
            x_out.flatten(start_dim=1),
            atom_type_one_hot_out.flatten(start_dim=1),
            charges_out.flatten(start_dim=1)
        ], dim=1)
        assert D_xt.shape == (B, n_nodes * (3+5+1))

        assert net_last_layer.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf)
        
        if self.detach_last_layer:
            net_last_layer = net_last_layer.detach()

        temb = get_timestep_embedding(ts*1000, self.temb_dim)
        temb = self.temb_net(temb) # (B, C)
        temb = temb.view(B, self.temb_dim, 1).repeat(1, 1, n_nodes) # (B, C, N)

        h = torch.cat([
            net_last_layer,
            atom_type_one_hot,
            charges.view(B, n_nodes, 1)
        ], dim=2)
        assert h.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf + 6)
        h = self.transformer_1_proj_in(h)
        assert h.shape == (B, n_nodes, self.transformer_dim)
        h = h.transpose(1,2)
        assert h.shape == (B, self.transformer_dim, n_nodes)

        for (res_block, attn_block) in zip(self.res_blocks, self.attn_blocks):
            h = res_block(h, temb)
            h = attn_block(h)

        h = h.transpose(1, 2)
        assert h.shape == (B, n_nodes, self.transformer_dim)

        rate_emb = self.pre_rate_proj(h) # (B, N, C)
        rate_emb = torch.mean(rate_emb, dim=1) # (B, C)
        rate_emb = self.post_rate_proj(rate_emb) # (B, rdim)

        if self.rate_use_x0_pred:
            x0_dim_logits = rate_emb
            rate_out = get_rate_using_x0_pred(
                x0_dim_logits=x0_dim_logits, xt_dims=st_batch.get_dims(),
                forward_rate=forward_rate, ts=ts, max_dim=st_batch.gs.max_problem_dim
            ).view(-1, 1) # (B, 1)
        else:
            x0_dim_logits = torch.zeros((B, st_batch.gs.max_problem_dim), device=device)
            f_rate_ts = forward_rate.get_rate(None, ts).view(B, 1)

            # rate_out = rate_emb.exp() # (B, 1)
            rate_out = F.softplus(rate_emb) * f_rate_ts # (B, 1)

        near_atom_logits = self.near_atom_proj(h)[:, :, 0]
        assert near_atom_logits.shape == (B, n_nodes)

        if sample_nearest_atom:
            if rnd is None:
                nearest_atom = torch.multinomial(torch.softmax(near_atom_logits, dim=1), 1).view(-1)
            else:
                nearest_atom = rnd.multinomial(torch.softmax(near_atom_logits, dim=1), num_samples=1).view(-1)

        assert nearest_atom.shape == (B,) # index from 0 to n_nodes-1

        # create a distance matrix for the closest atom (B, n_nodes)
        distances = torch.sum( (x[torch.arange(B, device=device), nearest_atom, :].view(B, 1, 3) - x)**2, dim=-1, keepdim=True).sqrt()
        assert distances.shape == (B, n_nodes, 1)

        nearest_atom_one_hot = torch.tensor([0.0, 1.0], device=device).view(1, 1, 2).repeat(B, n_nodes, 1)
        nearest_atom_one_hot[torch.arange(B, device=device), nearest_atom, 0] = 1.0
        nearest_atom_one_hot[torch.arange(B, device=device), nearest_atom, 1] = 0.0
        assert nearest_atom_one_hot.shape == (B, n_nodes, 2)

        vec_transformer_in = torch.cat([
            net_last_layer,
            atom_type_one_hot,
            charges.view(B, n_nodes, 1),
            distances,
            nearest_atom_one_hot
        ], dim=2)
        assert vec_transformer_in.shape == (B, n_nodes, self.egnn_net.egnn.hidden_nf + 6 + 1 + 2)
        vec_transformer_in = vec_transformer_in * node_mask
        vec_transformer_in = self.vec_transformer_in_proj(vec_transformer_in)
        assert vec_transformer_in.shape == (B, n_nodes, self.transformer_dim)
        vec_transformer_in = vec_transformer_in.transpose(1,2)
        assert vec_transformer_in.shape == (B, self.transformer_dim, n_nodes)
        h_vec = vec_transformer_in

        for (res_block, attn_block) in zip(self.vec_res_blocks, self.vec_attn_blocks):
            h_vec = res_block(h_vec, temb)
            h_vec = attn_block(h_vec)

        assert h_vec.shape == (B, self.transformer_dim, n_nodes)
        h_vec = h_vec.transpose(1, 2)
        assert h_vec.shape == (B, n_nodes, self.transformer_dim)

        vec_weights = self.vec_weighting_proj(h_vec) # (B, N, 1)
        assert vec_weights.shape == (B, n_nodes, 1)
        vectors = x[torch.arange(B, device=device), nearest_atom, :].view(B, 1, 3) - x
        assert vectors.shape == (B, n_nodes, 3)
        vectors = vectors * node_mask
        assert vectors.shape == (B, n_nodes, 3)
        # normalize the vectors
        vectors = vectors / (torch.sqrt(torch.sum(vectors**2, dim=-1, keepdim=True)) + 1e-3)

        auto_pos_mean_out = x[torch.arange(B, device=device), nearest_atom, :] + \
            torch.sum(vec_weights * vectors, dim=1) # (B, 3)

        pre_auto_h = self.pre_auto_proj(h_vec)
        assert pre_auto_h.shape == (B, n_nodes, self.transformer_dim)
        pre_auto_h = torch.mean(pre_auto_h, dim=1) # (B, C)
        post_auto_h = self.post_auto_proj(pre_auto_h) # (B, 2*5 + 2 + 1)

        pos_std = post_auto_h[:, 0:1].repeat(1, 3) # (B, 3)
        atom_type_mean = post_auto_h[:, 1:1+5] # (B, 5)
        atom_type_std = post_auto_h[:, 1+5:1+5+5] # (B, 5)
        charge_mean = post_auto_h[:, 1+5+5:1+5+5+1] # (B, 1)
        charge_std = post_auto_h[:, 1+5+5+1:1+5+5+1+1] # (B, 1)


        auto_mean_out = torch.cat(
            [auto_pos_mean_out, atom_type_mean, charge_mean],
        dim=1).view(B, 1, 3+5+1).repeat(1, n_nodes, 1) # (B, n_nodes, 3+5+1)
        auto_std_out = torch.cat(
            [pos_std, atom_type_std, charge_std],
        dim=1).view(B, 1, 3+5+1).repeat(1, n_nodes, 1) # (B, n_nodes, 3+5+1)

        auto_mean_out = torch.cat([
            auto_mean_out[:, :, 0:3].flatten(start_dim=1),
            auto_mean_out[:, :, 3:8].flatten(start_dim=1),
            auto_mean_out[:, :, 8:9].flatten(start_dim=1),
        ], dim=1) # (B, n_nodes * (3+5+1))

        auto_std_out = torch.cat([
            auto_std_out[:, :, 0:3].flatten(start_dim=1),
            auto_std_out[:, :, 3:8].flatten(start_dim=1),
            auto_std_out[:, :, 8:9].flatten(start_dim=1),
        ], dim=1) # (B, n_nodes * (3+5+1))

        auto_mask = st_batch.get_next_dim_added_mask(B, include_onehot_channels=True, include_obs=False) #(B, n_nodes * (3+5+1))

        auto_mean_out = auto_mask * auto_mean_out
        auto_std_out = auto_mask * auto_std_out
        
        return D_xt, rate_out, (auto_mean_out, auto_std_out), x0_dim_logits, near_atom_logits


EGNNMultiHeadJump_to_kwargs = {
    EGNNMultiHeadJump: set([
        ('detach_last_layer', 'str2bool', 'True'),
        ('rate_use_x0_pred', 'str2bool', 'False'),
        ('n_attn_blocks', 'int', 8),
        ('n_heads', 'int', 8),
        ('transformer_dim', 'int', 128),
    ])
}

