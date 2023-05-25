import os
import numpy as np
import torch

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

@torch.no_grad()
def edm_sampler(
    net, xT, y, randn_like=torch.randn_like,
    num_steps=200, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    return_intermediates=False, dim_deletion_process=None,
    min_dim=1, max_dim=float('inf'),
    x_cond=None, x_cond_mask=None,
    gradient_method_coef=0.0,
    constrain_endpoints=False,  # ensure that first frame is added first, last frame added second
    jump_net=None,
    use_per_dim_pred=False,  # only applicable for jump network trained with per_dim_pred
    plotty_plot=False,
):
    print('DOING SAMPLE!', min_dim, max_dim)

    random_id = np.random.randint(0, 2**32)
    plot_dir = f'plots/{random_id}'
    os.makedirs(plot_dir, exist_ok=True)
    print('PLOTTING TO', plot_dir)
    # for debugging: set random seed using randn_like to ensure reproducibility
    from scipy.stats import norm
    random_seed = int(norm.cdf(randn_like(xT).flatten()[0].item()) * 2**32)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    import random; random.seed(random_seed)
    print('Random seed:', random_seed)

    assert (x_cond is None) == (x_cond_mask is None)
    preaugment_structure = net.preaugment_structure
    jump_structure = net.structure

    dim_mask = tuple(t.unsqueeze(0) for t in jump_structure.dim_handler.which_dim_mask(device=xT.device))
    dim_mask = preaugment_structure.flatten_latents(dim_mask, contains_marg=True).squeeze(0)
    def pad_for_network_input(t, _net=net):
        padding_dims = _net.structure.latent_dim - preaugment_structure.latent_dim
        extras = torch.zeros_like(t[:, :1]).expand(-1, padding_dims)
        return torch.cat([t, extras], dim=1)

    def output_to_denoised_dimpred_indexpred(output, _jump_structure=jump_structure):
        data = _jump_structure.unflatten_batch(output, obs=y, pad_marg=False)
        denoised = data[:-2]
        dim_pred = torch.exp(data[-2])
        index_pred = torch.exp(data[-1])
        denoised = preaugment_structure.flatten_latents(denoised, contains_marg=False)
        return denoised, dim_pred, index_pred

    def sample_categorical(probs, dim=-1, sample_min=0, sample_max=float('inf'), is_recurse=False):
        probs = probs.clamp(min=0)
        if sample_min > 0:
            sl = [slice(None)] * len(probs.shape)
            sl[dim] = slice(None, sample_min)
            probs[sl] = 0
        if sample_max < probs.shape[dim]:
            sl = [slice(None)] * len(probs.shape)
            sl[dim] = slice(sample_max, None)
            probs[sl] = 0
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)
        try:
            cat = torch.distributions.Categorical(probs=probs)
        except ValueError:
            if is_recurse:
                raise Exception('Got nans in sample, even after resetting to uniform')
            print('WARNING: Failed to make categorical distribution. Resetting probs. They were:\n', probs)
            uniform_probs = torch.ones_like(probs)
            return sample_categorical(probs=uniform_probs, dim=dim, sample_min=sample_min, sample_max=sample_max, is_recurse=True)
        return cat.sample()

    if dim_deletion_process is not None:
        jump_diffusion = True
        highly_nonisotropic = dim_deletion_process.highly_nonisotropic
        assert net.pred_x0.lats[0, -1]
        init_xT = xT  # to use when adding dims
        dim_handler = jump_structure.dim_handler
        # convenient to just use max_sigma value from dim_deletion_process
        sigma_max = dim_deletion_process.large_value
        # set xT to initially have a single dimension
        B = xT.shape[0]
        dims = torch.zeros(B, dim_handler.max_dim, device=xT.device)
        dims[:, 0] = 1
        data_shaped_mask = dim_handler.batched_mask(dimses=dims)
        x_mask, y_mask = preaugment_structure.flatten_batch(data_shaped_mask, contains_marg=True)
        xT = xT * x_mask + (-1) * (1-x_mask)
        y = tuple(yi * yi_mask + (-1) * (1-yi_mask) for yi, yi_mask in zip(y, y_mask))
        y[1][:, 0] = 0  # set index to be zero
    else:
        jump_diffusion = False

    def get_sigmas(s):
        s_per_dim = dim_deletion_process.get_sigmas(s)
        # print('normil sigmas')
        # print(s_per_dim)
        return s_per_dim[dim_mask]

    def get_highly_nonisotropic_sigmas(s, ndims):
        s_per_dim = dim_deletion_process.get_highly_nonisotropic_sigmas(s, ndims)
        # print('get_highly_nonisotropic_sigmas')
        # print(s_per_dim)
        return s_per_dim[dim_mask]

    add_dim_rejected = torch.zeros(xT.shape[0], dtype=torch.long, device=xT.device)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=xT.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    # noise_mult = net.noise_mult.lats.to(xT.device)

    jump_per_index = dim_deletion_process.jump_per_index

    def gradient_method_prep_input(x):
        return x.detach().requires_grad_(gradient_method_coef > 0)

    def gradient_method_edit_denoised(denoised, x_cond, x_cond_mask, x, t_full):
        if gradient_method_coef > 0:
            error = (denoised - x_cond) * x_cond_mask
            l2 = error.pow(2).sum()
            # l2.backward()
            # grad = x.grad
            grad = torch.autograd.grad(l2, x)[0]
            denoised = denoised.detach() - gradient_method_coef * t_full * grad / 2
        return denoised

    all_ordered_frames = []

    # Main sampling loop.
    intermediates = {'denoised': [], 'xt': []} if return_intermediates else None
    x_next = xT.to(torch.float64) * t_steps[0] #* noise_mult
    highly_nonisotropic_dims = list(range(1, dim_handler.max_dim+1)) if jump_diffusion and highly_nonisotropic else [None]
    for highly_nonisotropic_dim in highly_nonisotropic_dims:
        # print('highly_nonisotropic_dim', highly_nonisotropic_dim)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            if plotty_plot and (t_cur == t_steps[t_steps>3].min() or t_next == 0):
                # add line to array of things to plot
                data = preaugment_structure.unflatten_batch(x_next, obs=y, pad_marg=False)
                frame_indices = data[2].flatten() # assume batch size 1
                frame_indices = frame_indices[frame_indices >= 0]
                ordered_frames = []
                frame_mapping = {p.long().item(): i for i, p in enumerate(frame_indices)}
                for frame_index in range(len(frame_indices)):
                    index_in_tensor = frame_mapping[frame_index]
                    ordered_frames.append(data[0][0][index_in_tensor].cpu().numpy().transpose(1, 2, 0).clip(-1, 1) * 127.5 + 127.5)
                ordered_frames = np.stack(ordered_frames).astype(np.uint8)
                all_ordered_frames.append(ordered_frames)

            if jump_diffusion and highly_nonisotropic:
                t_cur_full = get_highly_nonisotropic_sigmas(t_cur, ndims=highly_nonisotropic_dim)
                t_next_full = get_highly_nonisotropic_sigmas(t_next, ndims=highly_nonisotropic_dim)
            else:
                t_cur_full = get_sigmas(t_cur)
                t_next_full = get_sigmas(t_next)

            if x_cond is not None:
                # replace observed dimensions with observed values
                noised_x_cond = x_cond + randn_like(x_cond) * t_next_full.view(1, -1)
                x_cur = x_cur * (1 - x_cond_mask) + noised_x_cond * x_cond_mask

            # print('checking for nan 3')
            # print('x_cur', torch.isnan(x_cur).sum())
            # print('t_cur_full', torch.isnan(t_cur_full).sum())
            # print('t_next_full', torch.isnan(t_next_full).sum())
            # if torch.isnan(x_next).sum() > 0:
            #     exit()

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat_full = net.round_sigma((t_cur_full + gamma * t_cur_full))
            x_hat = x_cur + (t_hat_full ** 2 - t_cur_full ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            with torch.enable_grad():
                x_hat = gradient_method_prep_input(x_hat)
                # net.model.plot_path = f"{plot_dir}/test-net-input-{highly_nonisotropic_dim}-{round(t_cur.log().item()/3)}.png"
                #}.png"
                net_output = net(
                    x=pad_for_network_input(x_hat), y=y,
                    sigma_full=pad_for_network_input(t_hat_full.unsqueeze(0))
                    ).to(torch.float64)
                denoised, dim_pred, index_pred = output_to_denoised_dimpred_indexpred(net_output)
                denoised = gradient_method_edit_denoised(denoised, x_cond, x_cond_mask, x_hat, t_hat_full)
            if return_intermediates:
                intermediates['denoised'].append(denoised)
            d_cur = (x_hat - denoised) / t_hat_full
            x_next = x_hat + (t_next_full - t_hat_full) * d_cur
            t_is_zero = (t_hat_full == 0).unsqueeze(0)
            x_next[t_is_zero] = x_hat[t_is_zero]  # avoid nans when sigma is zero

            # print('checking for nan 4')
            # print('x_next', torch.isnan(x_next).sum())
            # print('x_hat', torch.isnan(x_hat).sum())
            # if torch.isnan(x_next).sum() > 0:
            #     exit()

            # Apply 2nd order correction.
            if i < num_steps - 1:
                # net.model.plot_path = f"{plot_dir}/test-net-input-{highly_nonisotropic_dim}-{round(t_cur.log().item()/3)}-corrector.png"
                net_output = net(
                    x=pad_for_network_input(x_next), y=y,
                    sigma_full=pad_for_network_input(t_next_full.unsqueeze(0))
                    ).to(torch.float64)
                denoised, _, _ = output_to_denoised_dimpred_indexpred(net_output)
                d_prime = (x_next - denoised) / t_next_full
                x_next = x_hat + (t_next_full - t_hat_full) * (0.5 * d_cur + 0.5 * d_prime)
                t_is_zero = (t_next_full == 0).unsqueeze(0)
                x_next[t_is_zero] = x_hat[t_is_zero]  # avoid nans when sigma is zero

            # print('checking for nan 5')
            # print('x_next', torch.isnan(x_next).sum())
            # print('x_hat', torch.isnan(x_hat).sum())
            # if torch.isnan(x_next).sum() > 0:
            #     exit()

            if return_intermediates:
                intermediates['xt'].append(x_next.type(torch.float32))

            if dim_deletion_process is not None and not highly_nonisotropic:

                if jump_net is not None:
                    # overwrite dim_pred and index_pred with jump_net predictions
                    jump_net_output = jump_net(
                        x=pad_for_network_input(x_hat, _net=jump_net), y=y,
                        sigma_full=pad_for_network_input(t_hat_full.unsqueeze(0), _net=jump_net)
                        ).to(torch.float64)
                    _, dim_pred, index_pred = output_to_denoised_dimpred_indexpred(
                        jump_net_output, _jump_structure=jump_net.structure,
                    )

                # Figure out if we are scheduled to add a new dimension
                dims_exist_cur = dim_deletion_process.get_sigmas(t_cur) < dim_deletion_process.large_value
                dims_exist_next = dim_deletion_process.get_sigmas(t_next) < dim_deletion_process.large_value
                n_additions_scheduled = dims_exist_next.sum() - dims_exist_cur.sum()
                assert (n_additions_scheduled <= 1).all(), "Only one dimension can be added at a time"
                if (not (n_additions_scheduled == 1).any()) or add_dim_rejected.all():
                    continue
                data = preaugment_structure.unflatten_batch(x_next, obs=y, pad_marg=False)
                current_dim = dim_handler.count_dims(data)

                print('adding dim at t={}'.format(t_next))

                # # if a new addition is scheduled, decide whether to do it
                # sample_min, sample_max = min_dim-1, max_dim
                # if constrain_endpoints:
                #     sample_min = max(sample_min, 1)
                # if jump_per_index:
                #     B, L, D = index_pred.shape
                #     added_dims_per_dim = sample_categorical(probs=index_pred, sample_min=0, sample_max=sample_max)
                #     for b in range(B):
                #         added_dims_per_dim[b, int(current_dim[b])+1:] = 0
                #     n_added_dims = added_dims_per_dim.sum(dim=1)
                #     x0_dims = current_dim + n_added_dims
                #     # get indices they are added at
                #     add_index_probs = added_dims_per_dim.clone().float()
                #     add_index_probs[:, 0] += (n_added_dims == 0).float()  # prevent all-zero probs if not adding dim
                #     add_index_probs = add_index_probs / add_index_probs.sum(dim=1, keepdim=True)
                #     jump_per_index_indices = sample_categorical(probs=add_index_probs)
                #     x0_dims = x0_dims.clamp(sample_min+1, sample_max+1)
                # else:
                #     x0_dims = sample_categorical(probs=dim_pred, sample_min=sample_min, sample_max=sample_max) + 1
                # add_dim = (n_additions_scheduled == 1) * (x0_dims > current_dim).float()
                # add_dim = add_dim * (1 - add_dim_rejected)  # don't add dim if previously rejected
                # addition_scheduled_and_rejected = (n_additions_scheduled == 1) * (1 - add_dim)
                # add_dim_rejected = (add_dim_rejected + addition_scheduled_and_rejected).clamp(0, 1)
                # # if add_dim_rejected.sum() > 0:
                # #     print('add dim rejected')
                # # print()

                # # sample where to add the dimensions, and add them
                # data = preaugment_structure.unflatten_batch(x_next, obs=y, pad_marg=False)
                # all_added_dims = torch.zeros((B, dim_handler.max_dim), device=x_next.device)
                # prev_n_dims = dim_handler.count_dims(data)
                # for b in range(B):
                #     if add_dim[b]:
                #         data_b = [t[b] for t in data]
                #         prev_n_dims_b = int(prev_n_dims[b])
                #         if constrain_endpoints:
                #             sample_min = 1
                #             sample_max = -1 if prev_n_dims > 1 else float('inf')
                #         if jump_per_index:
                #             add_index = jump_per_index_indices[b]
                #         else:
                #             add_index = sample_categorical(probs=index_pred[b:b+1, :prev_n_dims_b+1], sample_min=sample_min, sample_max=sample_max)[0]
                #         # print('adding dim', add_index)
                #         data_b, added_dims = dim_handler.add_dim(data_b, add_index=add_index)
                #         data = tuple(torch.cat([t[:b], tb.unsqueeze(0), t[b+1:]], dim=0) for t, tb in zip(data, data_b))
                #         all_added_dims[b] = added_dims
                #         # print('ADDING DIM!', 'at', t_next_full[0].item())
                # x_next, y = preaugment_structure.flatten_batch(data, contains_marg=True)
                # # set new latent dimensions to Gaussian noise
                # added_mask = dim_handler.batched_mask(all_added_dims)
                # added_mask_latent = preaugment_structure.flatten_latents(added_mask, contains_marg=True)
                # noise = t_next_full.view(1, -1) * randn_like(x_next)
                # x_next = x_next * (1 - added_mask_latent)  + noise * added_mask_latent

        if jump_diffusion and highly_nonisotropic:

            if jump_net is not None:
                jump_net.model.plot_path = f'{plot_dir}/test-jump-net-input-{highly_nonisotropic_dim}.png'
                print('plotting jump net input')
                # overwrite dim_pred and index_pred with jump_net predictions
                print('sigma_full', pad_for_network_input(t_next_full.unsqueeze(0), _net=jump_net))
                jump_net_output = jump_net(
                    x=pad_for_network_input(x_next, _net=jump_net), y=y,
                    sigma_full=pad_for_network_input(t_next_full.unsqueeze(0), _net=jump_net)
                    ).to(torch.float64)
                _, dim_pred, index_pred = output_to_denoised_dimpred_indexpred(
                    jump_net_output, _jump_structure=jump_net.structure,
                )

                if use_per_dim_pred:
                    index_pred = index_pred[:, :, :-1]
                #else:
                print('GOT INDEX PRED!', index_pred.shape)
                # plotting index_pred
                out_plot_path = f'{plot_dir}/test-jump-net-output-{highly_nonisotropic_dim}.png'
                if not os.path.exists(out_plot_path):
                    print('plotting jump net output to', out_plot_path)
                    import matplotlib.pyplot as plt
                    _B, _T, _D = index_pred.shape
                    fig, axes = plt.subplots(_B, _T, figsize=(_T, _B))
                    axes = np.array(axes).reshape(_B, _T)
                    for b in range(_B):
                        for t in range(_T):
                            # print('plotting', index_pred[b, t, :].detach().cpu().numpy())
                            # print(axes[b, t])
                            # print(list(range(_D)))
                            axes[b, t].bar(list(range(_D)), index_pred[b, t, :].detach().cpu().numpy())
                    plt.savefig(out_plot_path, bbox_inches='tight')

            # # check if things are nan
            # print('checking for nan 1')
            # print('x_hat', torch.isnan(x_hat).sum())
            # print('x_next', torch.isnan(x_next).sum())
            # print('dim_pred', torch.isnan(dim_pred).sum())
            # print('index_pred', torch.isnan(index_pred).sum())


            # figure out whether to add dimension
            index_min, index_max = 0, float('inf')
            # if constrain_endpoints:
            #     sample_min = max(sample_min, 1)
            #     index_min, index_max = (1, -1) if highly_nonisotropic_dim > 1 else (1, float('inf'))
            if constrain_endpoints and highly_nonisotropic_dim == 1:
                x0_dims = 2 * torch.ones((B,), device=x_next.device, dtype=torch.long)
                jump_per_index_indices = torch.ones((B,), device=x_next.device, dtype=torch.long)
            elif jump_per_index and use_per_dim_pred:
                B, L, D = index_pred.shape
                added_dims_per_dim = sample_categorical(probs=index_pred)
                if constrain_endpoints:
                    # precent adding to end
                    added_dims_per_dim[:, 0] = 0
                    added_dims_per_dim[:, highly_nonisotropic_dim] = 0
                for b in range(B):
                    added_dims_per_dim[b, highly_nonisotropic_dim+1:] = 0
                n_added_dims = added_dims_per_dim.sum(dim=1)
                # print()
                # print('ADDING NEW DIM')
                # print('added dims per dim', added_dims_per_dim)
                # print('n added dims', n_added_dims)
                x0_dims = highly_nonisotropic_dim + n_added_dims
                x0_dims = x0_dims  # .clamp(1, sample_max+1)

    # get indices they are added at
                # add_index_probs = added_dims_per_dim.clone().float()
                # add_index_probs += (n_added_dims == 0).float()  # prevent all-zero probs if not adding dim to prevent nans later
                add_index_probs = index_pred[:, :, 1:].sum(dim=2)
                add_index_probs[:, highly_nonisotropic_dim+1:] = 0
                if constrain_endpoints:
                    add_index_probs[:, 0] = 0
                    add_index_probs[:, highly_nonisotropic_dim] = 0
                add_index_probs = add_index_probs / add_index_probs.sum(dim=1, keepdim=True)

                # print('SAMPLNG ADD INDEX')
                # print('add_index_probs', add_index_probs)
                # print('sample_min', index_min)
                # print('sample_max', index_max)
                assert B == 1
                if x0_dims.item() < min_dim:
                    x0_dims = torch.ones_like(x0_dims) * min_dim
                if x0_dims.item() > max_dim:
                    x0_dims = torch.ones_like(x0_dims) * max_dim
                jump_per_index_indices = sample_categorical(probs=add_index_probs)
                # print('SAMPLED ADD INDEX', jump_per_index_indices)
                # print()
            elif jump_per_index:
                print('dim_pred', dim_pred)
                index_pred = index_pred[:, :highly_nonisotropic_dim+1, -1]
                print('index_pred', index_pred)
                x0_dims = sample_categorical(dim_pred, sample_min=max(highly_nonisotropic_dim-1, min_dim-1), sample_max=max_dim) + 1
            else:
                sample_min, sample_max = min_dim-1, max_dim
                x0_dims = sample_categorical(probs=dim_pred, sample_min=sample_min, sample_max=sample_max) + 1
            assert len(x0_dims) == 1, "Only implemented for batch size 1"
            add_dim = (x0_dims > highly_nonisotropic_dim).float().item()
            print('sampled x0 dims', x0_dims)
            # check for nans/inf
            print('added dim number', highly_nonisotropic_dim+1, ":", add_dim)
            if not add_dim:
                break
            # do the addition
            all_added_dims = torch.zeros((B, dim_handler.max_dim), device=x_next.device)
            data = preaugment_structure.unflatten_batch(x_next, obs=y, pad_marg=False)
            for b in range(B):
                data_b = [t[b] for t in data]
                if jump_per_index and (use_per_dim_pred or (constrain_endpoints and highly_nonisotropic_dim < 2)):  # good?
                    add_index = jump_per_index_indices[b]
                else:
                    index_min, index_max = (1, -1) if constrain_endpoints else (0, float('inf'))
                    print('sampling categorical for add index', index_pred[0:1, :highly_nonisotropic_dim+1], index_min, index_max)
                    add_index = sample_categorical(probs=index_pred[0:1, :highly_nonisotropic_dim+1], sample_min=index_min, sample_max=index_max)[0]
                print('adding dim at index', add_index)
                data_b, added_dims = dim_handler.add_dim(data_b, add_index=add_index)
                data = tuple(torch.cat([t[:b], tb.unsqueeze(0), t[b+1:]], dim=0) for t, tb in zip(data, data_b))
                all_added_dims[b] = added_dims
            x_next, y = preaugment_structure.flatten_batch(data, contains_marg=True)
            # set new latent dimensions to Gaussian noise
            added_mask = dim_handler.batched_mask(all_added_dims)
            added_mask_latent = preaugment_structure.flatten_latents(added_mask, contains_marg=True)
            noise = t_next_full.view(1, -1) * randn_like(x_next)
            x_next = x_next * (1 - added_mask_latent)  + noise * added_mask_latent

            # print('checking for nan 2')
            # print('x_hat', torch.isnan(x_hat).sum())
            # print('x_next', torch.isnan(x_next).sum())

    if plotty_plot:
        # plot all_ordered_frames
        # its a list of (varying length) numpy arrays of uint8 frames
        # pad with 255s to make them all the same length
        max_len = max([len(x) for x in all_ordered_frames])
        all_ordered_frames = [np.concatenate([x, 255 * np.ones((max_len - len(x), *x.shape[1:]), dtype=np.uint8)], axis=0) for x in all_ordered_frames]
        # stack frames within each along width dim
        all_ordered_frames = [np.concatenate(x, axis=1) for x in all_ordered_frames]
        # then stack along height dim
        all_ordered_frames = np.concatenate(all_ordered_frames, axis=0)
        # save as PNG
        from PIL import Image; from pathlib import Path
        Image.fromarray(all_ordered_frames).save(plotty_plot)
        print('saved to', plotty_plot)
    

            
    return intermediates if return_intermediates else (x_next.type(torch.float32), y)
