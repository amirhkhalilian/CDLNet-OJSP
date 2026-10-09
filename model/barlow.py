import torch
import torch.nn as nn

def sample_pixels(z, n):
    """ random subset of n pixel indices of codes z (B, M, H, W).
    Returns None (use all pixels) if n is None or n >= B*H*W.
    Use the same indices for both views to keep them pixel-aligned.
    """
    N = z.shape[0]*z.shape[2]*z.shape[3]
    if n is None or n >= N:
        return None
    return torch.randperm(N, device=z.device)[:n]

def codes_to_rows(z, idx=None):
    """ (B, M, H, W) codes -> (N, M) rows, one row per (b,h,w) pixel.
    """
    rows = z.permute(0,2,3,1).reshape(-1, z.shape[1])
    if idx is not None:
        rows = rows[idx]
    return rows

def barlow_loss(r1, r2, lambd=5e-3, eps=1e-5):
    """ Barlow Twins loss between pixel-aligned rows r1, r2 of shape (N, D).
    Each feature is standardized over N (as BatchNorm without affine),
    C = r1n^T r2n / N.
    on  = sum_i (1 - C_ii)^2
    off = sum_{i!=j} C_ij^2
    loss = (on + lambd*off) / D
    """
    N, D = r1.shape
    def standardize(r):
        return (r - r.mean(dim=0)) / torch.sqrt(r.var(dim=0, unbiased=False) + eps)
    C = standardize(r1).T @ standardize(r2) / N

    diag = torch.diagonal(C)
    on   = ((1 - diag)**2).sum()
    off  = (C**2).sum() - (diag**2).sum()
    loss = (on + lambd*off) / D
    return loss, on, off, C

class Projector(nn.Module):
    """ Barlow Twins projector, applied to per-pixel code vectors (N, M).
    """
    def __init__(self, M, dims=(2048, 2048, 2048)):
        super(Projector, self).__init__()
        layers = []
        d_in = M
        for d in dims[:-1]:
            layers += [nn.Linear(d_in, d, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True)]
            d_in = d
        layers.append(nn.Linear(d_in, dims[-1], bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, r):
        return self.layers(r)
