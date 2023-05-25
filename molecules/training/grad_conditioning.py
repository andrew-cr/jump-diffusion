import torch
import itertools as it
import numpy as np


class EDM():
    def __init__(self, lr_rampup_kimg):
        self.lr_rampup_kimg = lr_rampup_kimg

    def condition(self, optimizer, net, cur_nimg, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr * min(cur_nimg / max(self.lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)


EDM_to_kwargs = {
    EDM: set([
        ('lr_rampup_kimg', 'int', 10000),
    ])
}

class MoleculeJump():
    def __init__(self, lr_rampup_kimg, grad_norm_clip):
        self.lr_rampup_kimg = lr_rampup_kimg
        self.grad_norm_clip = grad_norm_clip

    def condition(self, optimizer, net, cur_nimg, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr * min(cur_nimg / max(self.lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_norm_clip)

MoleculeJump_to_kwargs = {
    MoleculeJump: set([
        ('lr_rampup_kimg', 'int', 320),
        ('grad_norm_clip', 'float', 10.0)
    ])
}



#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

class EGNN():
    def __init__(self):
        self.gradnorm_queue = Queue()
        self.gradnorm_queue.add(3000)  # Add large value that will be flushed.

    def condition(self, optimizer, net, cur_nimg, lr):

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()

        # Clips gradient and returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=max_grad_norm, norm_type=2.0)

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                f'while allowed {max_grad_norm:.1f}')

EGNN_to_kwargs = {
    EGNN: set([
    ])
}



grad_conditioners_to_kwargs = {
    l.__name__: kwargs for l, kwargs in it.chain(
        EDM_to_kwargs.items(),
        EGNN_to_kwargs.items(),
        MoleculeJump_to_kwargs.items(),
    )
}