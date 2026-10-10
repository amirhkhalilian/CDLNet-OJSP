#!/usr/bin/env python3
""" tests for model/metrics.py. run from repo root: python tests/test_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from model.metrics import xcorr, shift_coherence, dict_summary, net_dict_metrics, peak_frequency, freq_order, \
    effective_support, code_activity
from model.net import CDLNet, GDLNet

def brute_force_sc(W):
    """ shift-coherence via explicit loops over atom pairs and shifts.
    """
    W = W.numpy().astype(np.float64)
    W = W / np.linalg.norm(W.reshape(len(W),-1), axis=1)[:,None,None,None]
    M, C, P, _ = W.shape
    SC = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            for dy in range(-(P-1), P):
                for dx in range(-(P-1), P):
                    s = 0.0
                    for y in range(P):
                        for x in range(P):
                            if 0 <= y+dy < P and 0 <= x+dx < P:
                                s += (W[i,:,y+dy,x+dx]*W[j,:,y,x]).sum()
                    SC[i,j] = max(SC[i,j], abs(s))
    return SC

def test_brute_force():
    torch.manual_seed(0)
    W = torch.randn(4, 2, 5, 5)
    assert np.allclose(shift_coherence(W).numpy(), brute_force_sc(W), atol=1e-5)

def test_shifted_copy():
    torch.manual_seed(0)
    W = torch.zeros(2, 1, 11, 11)
    a = torch.randn(5, 5)
    W[0,0,1:6,2:7] = a
    W[1,0,4:9,1:6] = a  # shifted by (3,-1), fully inside support
    SC = shift_coherence(W)
    assert torch.allclose(SC, torch.ones(2,2), atol=1e-6)
    assert xcorr(W)[0,1,10,10].abs() < 0.99  # zero-shift sees them as different

def test_disjoint_channels():
    torch.manual_seed(0)
    W = torch.zeros(2, 2, 7, 7)
    W[0,0] = torch.randn(7,7)
    W[1,1] = torch.randn(7,7)
    SC = shift_coherence(W)
    assert SC[0,1] == 0 and SC[1,0] == 0
    assert torch.allclose(SC.diagonal(), torch.ones(2))

def test_scale_invariance():
    torch.manual_seed(0)
    W = torch.randn(6, 1, 7, 7)
    c = torch.tensor([3.0, -0.01, 1e3, -2.0, 0.5, 7.0])[:,None,None,None]
    assert torch.allclose(shift_coherence(W), shift_coherence(c*W), atol=1e-5)
    s1, s2 = dict_summary(W), dict_summary(c*W)
    assert all(abs(s1[k] - s2[k]) < 1e-5 for k in s1)

def test_symmetric_and_bounded():
    torch.manual_seed(0)
    W = torch.randn(10, 3, 7, 7)
    SC = shift_coherence(W)
    assert torch.allclose(SC, SC.T, atol=1e-6)
    assert SC.max() <= 1 + 1e-6
    assert torch.allclose(SC.diagonal(), torch.ones(10), atol=1e-6)

def test_dup_counts():
    torch.manual_seed(0)
    W = torch.zeros(5, 2, 11, 11)
    a = torch.randn(5, 5)
    W[0,0,0:5,0:5] = a
    W[1,0,6:11,3:8] = a        # shifted copy of atom 0
    W[2,0,2:7,2:7] = -2*a      # negated, scaled copy of atom 0
    W[3,1] = torch.randn(11,11)
    W[4,1] = torch.randn(11,11)
    s = dict_summary(W, thresh=0.9)
    assert shift_coherence(W)[3,4] < 0.9
    assert s["n_dup_pairs"] == 3, s
    assert s["n_dup_atoms"] == 3, s
    assert abs(s["mu_s"] - 1) < 1e-6

def test_net_dict_metrics():
    torch.manual_seed(0)
    for Net in [GDLNet, CDLNet]:
        net = Net(K=3, M=8, P=7, s=2, init=False)
        out = net_dict_metrics(net)
        assert len(out["A"]) == 3 and set(out["D"]) == set(out["A_mean"])
        # all layers start from the same filters
        assert all(abs(out["A"][k]["mu_s"] - out["A"][0]["mu_s"]) < 1e-6 for k in range(3))
        assert all(isinstance(v, (int, float)) for v in out["A_mean"].values())

def test_peak_frequency():
    P, n = 11, 64
    y, x = torch.meshgrid(torch.arange(P).float(), torch.arange(P).float(), indexing='ij')
    w = 2*np.pi*8/n
    W = torch.stack([torch.ones(P,P),             # DC
                     torch.cos(w*x),              # horizontal frequency -> theta 0
                     torch.cos(w*y),              # vertical frequency   -> theta pi/2
                     torch.cos(w*(x+y)/np.sqrt(2))])[:,None]  # diagonal -> theta pi/4
    radial, theta = peak_frequency(W, n=n)
    df = 2*np.pi/n
    assert radial[0] == 0
    assert all(abs(radial[i] - w) <= 1.5*df for i in [1,2,3]), radial
    assert abs(theta[1]) < 1e-6 or abs(theta[1] - np.pi) < 1e-6
    assert abs(theta[2] - np.pi/2) < 1e-6
    assert abs(theta[3] - np.pi/4) < 0.2, theta
    order = freq_order(W)
    assert order[0] == 0 and sorted(order) == [0,1,2,3]

def gauss_atom(P, sy, sx, theta=0.0):
    """ atom whose energy is a gaussian with std (sy, sx) pixels, rotated by theta.
    """
    y, x = torch.meshgrid(torch.arange(P).float(), torch.arange(P).float(), indexing='ij')
    y, x = y - (P-1)/2, x - (P-1)/2
    u =  np.cos(theta)*y + np.sin(theta)*x
    v = -np.sin(theta)*y + np.cos(theta)*x
    return torch.exp(-(u**2/(4*sy**2) + v**2/(4*sx**2)))  # squared -> energy std (sy, sx)

def test_effective_support():
    P = 21
    W = torch.stack([gauss_atom(P, 2.0, 0.7), gauss_atom(P, 0.7, 2.0), gauss_atom(P, 2.0, 0.7, np.pi/4)])[:,None]
    W = torch.cat([W, torch.zeros(1,1,P,P)])
    W[3,0,10,10] = 1                                 # spike
    smaj, smin = effective_support(W)
    assert torch.allclose(smaj[:3], torch.tensor(2.0), atol=0.02), smaj
    assert torch.allclose(smin[:3], torch.tensor(0.7), atol=0.02), smin  # rotation invariant
    assert smaj[3] == 0 and smin[3] == 0
    # scale and sign invariant
    assert torch.allclose(effective_support(-3*W)[0], smaj)

def test_collapsed_counts():
    torch.manual_seed(0)
    P = 11
    W = torch.zeros(6, 1, P, P)
    W[0,0,5,5] = 1; W[1,0,5,5] = -2                  # two spikes: collapsed duplicates
    W[2,0] = gauss_atom(P, 0.5, 0.5)                 # small blob: collapsed
    W[3,0] = gauss_atom(P, 2.0, 0.4)                 # thin line: not collapsed (long on one axis)
    W[4,0] = gauss_atom(P, 1.5, 1.5)*torch.cos(torch.arange(P).float())[None,:]  # gabor
    W[5,0] = -W[4,0]                                 # duplicate of the gabor
    s = dict_summary(W, thresh=0.9, collapse_px=0.8)
    assert s["n_collapsed"] == 3, s
    assert s["n_dup_pairs_nc"] == 1 and s["n_dup_atoms_nc"] == 2, s  # only the gabor pair
    assert s["n_dup_pairs"] >= 2  # spikes + gabor pair (+ spike/blob)

def test_code_activity():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, init=False)
    with torch.no_grad():
        net.t.zero_(); net.t[:, 0, 3] = 1e6          # subband 3 thresholded away at every iteration
    # a real DataLoader: iterating it draws a base seed from the global RNG
    loader = torch.utils.data.DataLoader(torch.rand(3, 1, 32, 32), batch_size=2, shuffle=False)
    rng = torch.random.get_rng_state()
    act = code_activity(net, loader, sigma=25)
    assert torch.equal(rng, torch.random.get_rng_state())  # global RNG untouched
    assert torch.equal(act, code_activity(net, loader, sigma=25))  # deterministic
    assert act.shape == (3, 8) and torch.all(act[:, 3] == 0)
    assert torch.all(act[:, [0,1,2,4,5,6,7]] > 0.9)  # zero thresholds: (almost) all coefficients active
    m = net_dict_metrics(net, activity=act)
    assert all(a["n_dead"] == 1 for a in m["A"]) and m["D"]["n_dead"] == 1 and m["A_mean"]["n_dead"] == 1
    assert "n_dead" not in net_dict_metrics(net)["A_mean"]

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"PASS {name}")
    print(f"all {len(tests)} tests passed.")
