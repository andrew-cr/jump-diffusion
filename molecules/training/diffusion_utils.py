import torch
import math
import numpy as np
from tqdm import tqdm



# Class for DDPM continuous time related statistics
class VP_SDE():
    def __init__(self, max_dim, beta_min, beta_max):
        super().__init__()
        self.max_dim = max_dim

        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta_t(self, ts):
        return (ts * self.beta_max + (1-ts) * self.beta_min).view(-1, 1).repeat(1, self.max_dim)

    def get_sigma(self, times):
        """
            returns sqrt(1-alpha_bar_t) in DDPM/SDE notation 
        """
        log_term = -0.25 * times ** 2 * (self.beta_max - self.beta_min) - 0.5 * times * self.beta_min
        alpha_squared = torch.exp(2*log_term)
        return torch.sqrt(1 - alpha_squared)

    def get_p0t_stats(self, st_batch, times):
        # minibatch (batch, dim1, dim2, ..., dimD)
        # times (batch)
        minibatch = st_batch.get_flat_lats()

        log_term = -0.25 * times ** 2 * (self.beta_max - self.beta_min) - 0.5 * times * self.beta_min
        log_term_unsqueezed = log_term.view(
            minibatch.shape[0],
            *([1] * (len(minibatch.shape)-1))
        )
        mean = torch.exp(log_term_unsqueezed) * minibatch
        std = torch.sqrt(1 - torch.exp(2. * log_term_unsqueezed)).expand(*minibatch.shape)

        return mean, std

    def predict_x0_from_xt(self, xt, eps, t):
        # xt (B, D)
        # eps (B, D)
        # t (B)
        log_term = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_term_unsqueezed = log_term.view(
            xt.shape[0],
            *([1] * (len(xt.shape)-1))
        )
        std = torch.sqrt(1 - torch.exp(2. * log_term_unsqueezed)).expand(*xt.shape)

        return (xt - std * eps) / torch.exp(log_term_unsqueezed)

    def predict_eps_from_x0_xt(self, xt_st_batch, x0, t):
        xt = xt_st_batch.get_flat_lats()
        log_term = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_term_unsqueezed = log_term.view(
            xt.shape[0],
            *([1] * (len(xt.shape)-1))
        )
        std = torch.sqrt(1 - torch.exp(2. * log_term_unsqueezed)).expand(*xt.shape)

        return (xt - torch.exp(log_term_unsqueezed) * x0) / std

    def get_pxt2_xt1_stats(self, xt1_st_batch, t1, t2):
        # p(x_t2 | x_t1) gaussian stats

        minibatch = xt1_st_batch.get_flat_lats() # (B, N)

        alpha_t1 = torch.exp(-0.5 * t1**2 * (self.beta_max - self.beta_min) - t1 * self.beta_min)
        alpha_t2 = torch.exp(-0.5 * t2**2 * (self.beta_max - self.beta_min) - t2 * self.beta_min)
        alpha_t1 = alpha_t1.view(-1, 1)
        alpha_t2 = alpha_t2.view(-1, 1)

        mean = torch.sqrt(alpha_t2 / alpha_t1) * minibatch
        std = torch.sqrt(1 - alpha_t2 / alpha_t1).repeat(1, minibatch.shape[1])

        return mean, std








class ForwardRate():
    def get_rate(self, dims, ts):
        """
            Gets the rate evaluated at times ts (B,) 
            Dims is ignored
        """
        raise NotImplementedError



class StateIndependentForwardRate(ForwardRate):
    """
        Generic class representing a state independent forward rate function 
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim
        self.max_num_deletions = self.max_dim - 1
        self.std_mult = 0.7
        self.offset = 0.1

        # scaling of the rate function is such that the mean number of deletions
        # is std_mult standard deviations above c so that a good proportion of
        # trajectories will reach the maximum number of deletions during the
        # forward process

        # add a small offset so we never have 0 rate which may have problems with
        # optimization

    def get_rate_integral(self, ts):
        """
            Gets the integral of the rate between time 0 and ts (B,) 
        """
        raise NotImplementedError

    def get_dims_at_t(self, start_dims, ts):
        dims_deleted_at_t = torch.poisson(self.get_rate_integral(ts))
        dims_xt = (start_dims - dims_deleted_at_t).clamp(min=1).int()
        return dims_xt

    def get_dims_at_t2_starting_t1(self, dims_t1, t1, t2):
        integral = self.get_rate_integral(t2) - self.get_rate_integral(t1)
        dims_deleted = torch.poisson(integral)
        dims_t2 = (dims_t1 - dims_deleted).clamp(min=1).int()
        return dims_t2




class StepForwardRate(StateIndependentForwardRate):
    def __init__(self, max_dim, rate_cut_t):
        super().__init__(max_dim)
        self.rate_cut_t = rate_cut_t
        assert self.rate_cut_t > 0
        assert self.rate_cut_t < 1

    def get_scalar(self):
        T = self.rate_cut_t # the step change point
        c = self.max_num_deletions
        return (2 * (1-T) * c + self.std_mult**2 * (1-T) + math.sqrt((-2 * (1-T) * c - self.std_mult**2 * (1-T))**2 - 4 * (1-T)**2 * c**2   )) / (2 * (1-T)**2 )

    def get_rate(self, dims, ts):
        T = self.rate_cut_t
        return self.get_scalar() * (ts > T).long() + self.offset

    def get_rate_integral(self, ts):
        T = self.rate_cut_t
        return (ts - T) * self.get_scalar() * (ts > T).long() + self.offset * ts



class ConstForwardRate(StateIndependentForwardRate):
    def __init__(self, max_dim, scalar=None):
        super().__init__(max_dim)
        self.scalar = scalar

    def get_scalar(self):
        try:
            if self.scalar is None:
                c = self.max_num_deletions
                return (2 * c + self.std_mult**2 + math.sqrt((self.std_mult**2 + 2 * c)**2 - 4 * c**2)) / 2
            else:
                return self.scalar
        except AttributeError:
            print("ConstForwardRate: scalar not set. Presumably because old checkpoint was loaded. Reverting to old method. TODO delete this exception later.")
            c = self.max_num_deletions
            return (2 * c + self.std_mult**2 + math.sqrt((self.std_mult**2 + 2 * c)**2 - 4 * c**2)) / 2

    def get_rate(self, dims, ts):
        return self.get_scalar() * torch.ones_like(ts)

    def get_rate_integral(self, ts):
        return self.get_scalar() * ts









def get_rate_using_x0_pred(x0_dim_logits, xt_dims, forward_rate, ts, max_dim):
    # Note this assumes a State Independent Forward Rate
    assert isinstance(forward_rate, StateIndependentForwardRate)
    B = x0_dim_logits.shape[0]

    device = x0_dim_logits.device
    assert x0_dim_logits.shape == (B, max_dim)

    # zero out all probability for x0dim < xt_dim which is impossible
    x0_dim_probs = torch.zeros_like(x0_dim_logits)
    for idx in range(B):
        # if xt_dims[idx] == max_dim then only allow [-1:]
        # if xt_dims[idx] == max_dim - 1 then only allow [-2:]
        # ...
        # only allow through [xt_dims[idx] - max_dim - 1 : ]
        idx_start = xt_dims[idx] - max_dim - 1
        x0_dim_probs[idx, idx_start:] = torch.softmax(x0_dim_logits[idx, idx_start:], dim=0)

    # calculation is
    # rev_rate = f_rate \sum_{d_x0} ( p(d_x + 1 | d_x0) / p(d_x | d_x0) ) * p(d_x0 | x)
    # x0_dim_probs = p(d_x0 | x) shape (B, max_dim)
    # ratios = p(d_x + 1 | d_x0) / p(d_x | d_x0) shape (B, max_dim)
    dx0range = torch.arange(1, max_dim+1, device=device)
    ratios = torch.zeros((B, max_dim), device=device)
    f_rate_integrals = forward_rate.get_rate_integral(ts) # (B,)
    
    truncation = max_dim*2
    dx0_truncation_array = torch.arange(0, truncation, device=device).view(1, truncation) + \
        torch.arange(0, max_dim, device=device).view(max_dim, 1)

    for idx in range(B):
        if xt_dims[idx] > 1:
            ratios[idx, :] = (1/f_rate_integrals[idx]) * (dx0range - xt_dims[idx])
            ratios[idx, ratios[idx, :] < 0] = 0.0
        else:
            dim1_presum_logprobs = torch.distributions.poisson.Poisson(f_rate_integrals[idx]).log_prob(dx0_truncation_array)
            assert dim1_presum_logprobs.shape == (max_dim, truncation)
            dim1_logprobs = torch.logsumexp(dim1_presum_logprobs, dim=1) # (max_dim,)

            dim2_logprobs = torch.distributions.poisson.Poisson(f_rate_integrals[idx]).log_prob(
                torch.arange(-1, max_dim-1, device=device).clamp(min=0)
            )
            assert dim2_logprobs.shape == (max_dim,)
            # the first element is the probability of x_t dim 2 whilst x_0 dim =1 so set to zero
            dim2_logprobs[0] = -1000

            ratios[idx, :] = torch.exp(dim2_logprobs - dim1_logprobs)

            ratios[idx, dx0range < xt_dims[idx]] = 0.0

    return forward_rate.get_rate(dims=None, ts=ts) * torch.sum(ratios * x0_dim_probs, dim=1)