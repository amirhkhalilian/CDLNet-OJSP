#!/usr/bin/env python3
""" tests for model/metrics.py. run from repo root: python tests/test_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from model.metrics import xcorr, shift_coherence, dict_summary, net_dict_metrics
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

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"PASS {name}")
    print(f"all {len(tests)} tests passed.")
