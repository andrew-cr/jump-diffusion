import itertools as it
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import StructuredDatasetBase


class MatrixFactorizationDataset(StructuredDatasetBase):
    """
    Returns A, R, a tensor of intermediate products, and E=AR, 
    """
    name = "matrix_factorization"
    is_image = [0, 0, 0, 0]
    is_onehot = None  # set in __init__
    names = ["A", "R", "C", "E"]

    def __init__(self, matrix_n, matrix_m, matrix_k, factors_binary, bernoulli_p, max_size=None):
        self.n = matrix_n
        self.m = matrix_m
        self.k = matrix_k
        self.A_binary, self.R_binary = factors_binary
        self.bernoulli_p = bernoulli_p
        self.is_onehot = [factors_binary[0], factors_binary[1], 0, 0]

    def __len__(self):
        return 10**12  # interpreted as infinity by misc.InfiniteSampler

    def __getitem__(self, index, will_augment=True, deterministic=False):
        assert not will_augment
        rnd = torch.Generator('cpu').manual_seed(index)
        def rand(*args, **kwargs):
            return torch.rand(*args, **kwargs, generator=rnd)
        A = rand(self.n, self.k)
        R = rand(self.k, self.m)
        if self.A_binary:
            A = (A < self.bernoulli_p).to(torch.float32)
        if self.R_binary:
            R = (R < self.bernoulli_p).to(torch.float32)
        C = torch.einsum('nk,km->nkm', A, R)
        E = torch.einsum('nkm->nm', C)
        if self.A_binary:
            A = torch.stack([1-A, A], dim=0)
        if self.R_binary:
            R = torch.stack([1-R, R], dim=0)
        return A, R, C, E

    def get_graphical_model_edges(self):
        numels = [self.n*self.k, self.k*self.m, self.n*self.k*self.m, self.n*self.m]
        numels_before = lambda name: sum(numels[:self.names.index(name)])
        index_A = lambda i, j: numels_before('A') + i*self.k + j
        index_R = lambda j, k: numels_before('R') + j*self.m + k
        index_C = lambda i, j, k: numels_before("C") + i*self.k*self.m + j*self.m + k
        index_E = lambda i, j: numels_before("E") + i*self.m + j
        edges = set()
        for i, j, k in it.product(range(self.n), range(self.m), range(self.k)):
            edges.add((index_A(i, k), index_C(i, j, k)))
            edges.add((index_R(k, j), index_C(i, j, k)))
            edges.add((index_C(i, j, k), index_E(i, j)))
        edges = set.union(edges, set([(b, a) for a, b in edges]))  # symmetrize
        return edges

    def get_graphical_model_mask_and_indices(self):
        # Compute an alternative representation of graphical model edges that is more convenient for the
        # the sparse attention mechanism.
        numels = [self.n*self.k, self.k*self.m, self.n*self.k*self.m, self.n*self.m]
        numels_before = lambda name: sum(numels[:self.names.index(name)])
        index_A = lambda i, j: numels_before('A') + i*self.k + j
        index_R = lambda j, k: numels_before('R') + j*self.m + k
        index_C = lambda i, j, k: numels_before("C") + i*self.k*self.m + j*self.m + k
        index_E = lambda i, j: numels_before("E") + i*self.m + j
        attendable_indices = [set([i]) for i in range(sum(numels))]
        for n, k, m in it.product(range(self.n), range(self.k), range(self.m)):
            attendable_indices[index_A(n, k)].add(index_C(n, k, m))
            attendable_indices[index_C(n, k, m)].add(index_A(n, k))
            attendable_indices[index_R(k, m)].add(index_C(n, k, m))
            attendable_indices[index_C(n, k, m)].add(index_R(k, m))
            attendable_indices[index_E(n, m)].add(index_C(n, k, m))
            attendable_indices[index_C(n, k, m)].add(index_E(n, m))
        attendable_indices = [list(indices) for indices in attendable_indices]
        max_attendable_indices = max(len(indices) for indices in attendable_indices)
        mask = torch.zeros(sum(numels), max_attendable_indices, dtype=torch.float32)
        for i, indices in enumerate(attendable_indices):
            mask[i, :len(indices)] = 1
        attendable_indices = [indices + [0]*(max_attendable_indices - len(indices)) for indices in attendable_indices]
        attendable_indices = torch.tensor(attendable_indices, dtype=torch.long)
        return mask, attendable_indices

    def get_shareable_embedding_indices(self):
        n, m, k = self.n, self.m, self.k
        return torch.cat([i * torch.ones(numel, dtype=torch.long) for i, numel in zip([0, 1, 2, 3], [n*k, k*m, n*k*m, n*m])])

    def augment(self, data, augment_pipe):
        assert augment_pipe is None, "No augmentation for matrix factorization dataset."
        return data, None

    def data_to_A_R_Ehat_E(self, data):
        A, R, C, E = data
        device = next(t.device for t in data if t is not None)
        A = A if A is not None else torch.zeros(1, 2, self.n, self.k, device=device)
        R = R if R is not None else torch.zeros(1, 2, self.k, self.m, device=device)
        E = E if E is not None else torch.zeros(1, self.n, self.m, device=device)
        discretize = lambda x: x.argmax(dim=1).float()
        A = discretize(A) if self.A_binary else A
        R = discretize(R) if self.R_binary else R
        E_hat = torch.einsum('bnk,bkm->bnm', A, R)
        return A.cpu(), R.cpu(), E_hat.cpu(), E.cpu()

    def log_batch(self, data):
        d = super().log_batch(data, return_dict=True)
        A, R, E_hat, E = self.data_to_A_R_Ehat_E(data)
        batch_size = A.shape[0]
        baseline_rmse = torch.sqrt(torch.mean((E - E.mean(dim=0, keepdims=True))**2)) * batch_size / (batch_size - 1)
        rmse = torch.sqrt(torch.mean((E - E_hat)**2))
        d['MatrixFactorization/baseline_rmse'] = baseline_rmse.item()
        d['MatrixFactorization/rmse'] = rmse.item()
        fig = self.plot(data)
        d['Images/Reconstruction'] = wandb.Image(fig)
        wandb.log(d)

    def plot(self, data):
        A, R, E_hat, E = self.data_to_A_R_Ehat_E(data)
        batch_size = min(A.shape[0], 4)
        fig, axes = plt.subplots(nrows=batch_size, ncols=4)
        axes[0, 0].set_title('A')
        axes[0, 1].set_title('R')
        axes[0, 2].set_title('Ê')
        axes[0, 3].set_title('E')
        for ax_row, a, r, e_hat, e in zip(axes, A, R, E_hat, E):
            kwargs = {'vmin': 0., 'vmax': E.max(), 'cmap': 'binary'}
            ax_row[0].imshow(a, **kwargs)
            ax_row[1].imshow(r, **kwargs)
            ax_row[2].imshow(e_hat, **kwargs)
            ax_row[3].imshow(e, **kwargs)
        for ax in np.array(axes).flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        return fig


matrix_factorization_datasets_to_kwargs = {
    MatrixFactorizationDataset: set([
        ('factors_binary', 'parse_int_list', (0, 0)),
        ('matrix_n', 'int', 16),
        ('matrix_m', 'int', 16),
        ('matrix_k', 'int', 8),
        ('bernoulli_p', 'float', 0.3),
    ]),
}
matrix_factorization_kwargs_gettable_from_dataset = {MatrixFactorizationDataset: []}