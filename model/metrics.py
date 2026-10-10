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

def effective_support(W, eps=1e-12):
    """ spatial std (pixels) of each atom's energy along its major and minor axis,
    from the eigenvalues of the energy-weighted spatial covariance (summed over channels).
    W: (M, C, P, P) filterbank
    output: std_major, std_minor, each of shape (M,)
    """
    e = (W**2).sum(dim=1)
    e = e / e.flatten(1).sum(dim=1).clamp(min=eps)[:,None,None]
    i = torch.arange(W.shape[-1], device=W.device, dtype=W.dtype)
    y, x = torch.meshgrid(i, i, indexing='ij')
    dy = y - (e*y).sum(dim=(1,2))[:,None,None]
    dx = x - (e*x).sum(dim=(1,2))[:,None,None]
    cyy, cxx, cxy = (e*dy*dy).sum(dim=(1,2)), (e*dx*dx).sum(dim=(1,2)), (e*dx*dy).sum(dim=(1,2))
    half_tr = (cyy + cxx)/2
    disc = torch.sqrt((half_tr**2 - (cyy*cxx - cxy**2)).clamp(min=0))
    return torch.sqrt(half_tr + disc), torch.sqrt((half_tr - disc).clamp(min=0))

def dict_summary(W, thresh=0.9, collapse_px=0.8):
    """ redundancy summary of filterbank W (M, C, P, P).
    mu_s, mean_s:   max, mean off-diagonal shift-coherence
    mu_0, mean_0:   max, mean off-diagonal zero-shift coherence
    n_dup_pairs:    num. pairs i<j with SC_ij > thresh
    n_dup_atoms:    num. atoms with at least one such partner
    n_collapsed:    num. atoms with energy std < collapse_px along every axis
    n_dup_pairs_nc, n_dup_atoms_nc: as above, among non-collapsed atoms only
    """
    R  = xcorr(W).abs()
    P  = W.shape[-1]
    SC = R.amax(dim=(2,3))
    Z  = R[:,:,P-1,P-1]

    M = SC.shape[0]
    offdiag = ~torch.eye(M, dtype=torch.bool, device=SC.device)
    dup = (SC > thresh) & offdiag
    collapsed = effective_support(W)[0] < collapse_px
    dup_nc = dup & ~collapsed[:,None] & ~collapsed[None,:]

    return {"mu_s":           SC[offdiag].max().item(),
            "mean_s":         SC[offdiag].mean().item(),
            "mu_0":           Z[offdiag].max().item(),
            "mean_0":         Z[offdiag].mean().item(),
            "n_dup_pairs":    dup.triu().sum().item(),
            "n_dup_atoms":    dup.any(dim=1).sum().item(),
            "n_collapsed":    collapsed.sum().item(),
            "n_dup_pairs_nc": dup_nc.triu().sum().item(),
            "n_dup_atoms_nc": dup_nc.any(dim=1).sum().item()}

@torch.no_grad()
def code_activity(net, loader, sigma=25, seed=0):
    """ fraction of nonzero sparse-code coefficients per iteration k and subband j, (K, M),
    averaged over the images of loader with AWGN of level sigma (on [0,255]).
    Noise comes from a local fixed-seed generator (deterministic), and the global CPU RNG,
    which a DataLoader draws its base seed from, is restored afterwards: training is unaffected.
    """
    device = next(net.parameters()).device
    g = torch.Generator().manual_seed(seed)
    act, n = 0, 0
    with torch.random.fork_rng(devices=[]):
        for x in loader:
            x = x.to(device)
            y = x + (sigma/255)*torch.randn(x.shape, generator=g).to(device)
            s = torch.full((x.shape[0],1,1,1), float(sigma), device=device)
            zs = list(net.forward_generator(y, s))[:-1]
            act = act + torch.stack([(z != 0).float().mean(dim=(2,3)).sum(dim=0) for z in zs])
            n += x.shape[0]
    return act / n

@torch.no_grad()
def net_dict_metrics(net, thresh=0.9, collapse_px=0.8, activity=None, dead_thresh=1e-3):
    """ dict_summary of each analysis filterbank A_k and of the dictionary D.
    A_mean holds the mean over k of each A_k summary value.
    activity: optional (K, M) code_activity; adds n_dead (subbands active on < dead_thresh
    of code coefficients) for each A_k (code z^k) and for D (final code).
    """
    A = [dict_summary(get_filters(net.A[k]), thresh, collapse_px) for k in range(net.K)]
    D = dict_summary(get_filters(net.D), thresh, collapse_px)
    if activity is not None:
        for k in range(net.K):
            A[k]["n_dead"] = (activity[k] < dead_thresh).sum().item()
        D["n_dead"] = (activity[-1] < dead_thresh).sum().item()
    A_mean = {key: float(np.mean([a[key] for a in A])) for key in A[0]}
    return {"thresh": thresh, "collapse_px": collapse_px, "dead_thresh": dead_thresh,
            "A_mean": A_mean, "A": A, "D": D}
