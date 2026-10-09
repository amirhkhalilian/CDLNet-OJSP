import numpy as np
import torch
import torch.nn.functional as F

def get_filters(op):
    """ filterbank of a (Gabor) conv operator as a tensor of shape (M, C, P, P).
    """
    if hasattr(op, 'get_filter'):
        W = op.get_filter()
    else:
        W = op.weight
    return W.detach()

def xcorr(W, eps=1e-12):
    """ cross-correlation between all pairs of unit-norm atoms at all shifts.
    W: (M, C, P, P) filterbank
    output: (M, M, 2P-1, 2P-1), output[i,j,tau] = sum_{c,u} w_i[c,u+tau] w_j[c,u]
    """
    W = W / W.flatten(1).norm(dim=1).clamp(min=eps)[:,None,None,None]
    P = W.shape[-1]
    return F.conv2d(W, W, padding=P-1)

def shift_coherence(W):
    """ shift-coherence matrix, SC_ij = max_tau |<w_i, S_tau w_j>| for unit-norm atoms.
    W: (M, C, P, P) filterbank
    output: (M, M)
    """
    return xcorr(W).abs().amax(dim=(2,3))

def peak_frequency(W, n=64):
    """ frequency at the peak of each atom's magnitude response (summed over channels).
    W: (M, C, P, P) filterbank
    output: radial frequency in rad/pixel, orientation in [0, pi), each of shape (M,)
    """
    X = torch.fft.fft2(W, s=(n,n)).abs().sum(dim=1)
    k = X.flatten(1).argmax(dim=1)
    f = 2*np.pi*torch.fft.fftfreq(n, device=W.device)
    wy, wx = f[k // n], f[k % n]
    return torch.sqrt(wy**2 + wx**2), torch.atan2(wy, wx) % np.pi

def freq_order(W, n_bins=6):
    """ atom ordering by peak radial frequency (binned), then orientation.
    """
    radial, theta = [v.cpu().numpy() for v in peak_frequency(W)]
    rbin = np.floor(radial / (np.pi*np.sqrt(2)/n_bins))
    return np.lexsort((theta, rbin))

def dict_summary(W, thresh=0.9):
    """ redundancy summary of filterbank W (M, C, P, P).
    mu_s, mean_s: max, mean off-diagonal shift-coherence
    mu_0, mean_0: max, mean off-diagonal zero-shift coherence
    n_dup_pairs:  num. pairs i<j with SC_ij > thresh
    n_dup_atoms:  num. atoms with at least one such partner
    """
    R  = xcorr(W).abs()
    P  = W.shape[-1]
    SC = R.amax(dim=(2,3))
    Z  = R[:,:,P-1,P-1]

    M = SC.shape[0]
    offdiag = ~torch.eye(M, dtype=torch.bool, device=SC.device)
    dup = (SC > thresh) & offdiag

    return {"mu_s":        SC[offdiag].max().item(),
            "mean_s":      SC[offdiag].mean().item(),
            "mu_0":        Z[offdiag].max().item(),
            "mean_0":      Z[offdiag].mean().item(),
            "n_dup_pairs": dup.triu().sum().item(),
            "n_dup_atoms": dup.any(dim=1).sum().item()}

@torch.no_grad()
def net_dict_metrics(net, thresh=0.9):
    """ dict_summary of each analysis filterbank A_k and of the dictionary D.
    A_mean holds the mean over k of each A_k summary value.
    """
    A = [dict_summary(get_filters(net.A[k]), thresh) for k in range(net.K)]
    D = dict_summary(get_filters(net.D), thresh)
    A_mean = {key: float(np.mean([a[key] for a in A])) for key in A[0]}
    return {"thresh": thresh, "A_mean": A_mean, "A": A, "D": D}
