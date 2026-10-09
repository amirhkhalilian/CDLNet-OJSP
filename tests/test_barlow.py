#!/usr/bin/env python3
""" tests for model/barlow.py. run from repo root: python tests/test_barlow.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.barlow import sample_pixels, codes_to_rows, barlow_loss, Projector

def test_identical_views():
    torch.manual_seed(0)
    r = torch.randn(4096, 16)
    loss, on, off, C = barlow_loss(r, r)
    assert on < 1e-6, on
    assert torch.allclose(C, C.T, atol=1e-5)

def test_independent_views():
    torch.manual_seed(0)
    N, D = 20000, 16
    loss, on, off, C = barlow_loss(torch.randn(N, D), torch.randn(N, D))
    assert abs(on.item() - D) < 0.1*D, on            # C_ii ~ 0
    assert abs(off.item() - D*(D-1)/N) < 0.5*D*(D-1)/N, off  # E[C_ij^2] = 1/N

def test_affine_invariance():
    torch.manual_seed(0)
    r1 = torch.randn(1000, 8)
    r2 = r1 + 0.5*torch.randn(1000, 8)
    a1, b1 = torch.rand(8) + 0.1, torch.randn(8)
    a2, b2 = torch.rand(8) + 0.1, torch.randn(8)
    l, *_ = barlow_loss(r1, r2, eps=1e-12)
    m, *_ = barlow_loss(a1*r1 + b1, a2*r2 + b2, eps=1e-12)
    assert torch.allclose(l, m, rtol=1e-5), (l, m)

def test_loss_value():
    torch.manual_seed(0)
    r1, r2 = torch.randn(500, 6), torch.randn(500, 6)
    loss, on, off, C = barlow_loss(r1, r2, lambd=0.1)
    assert torch.allclose(loss, (on + 0.1*off)/6)
    I = torch.eye(6, dtype=torch.bool)
    assert torch.allclose(on, ((1-C[I])**2).sum()) and torch.allclose(off, (C[~I]**2).sum())

def test_gradcheck():
    torch.manual_seed(0)
    z1 = torch.randn(2, 4, 3, 3, dtype=torch.double, requires_grad=True)
    z2 = torch.randn(2, 4, 3, 3, dtype=torch.double, requires_grad=True)
    idx = sample_pixels(z1, 10)
    f = lambda a, b: barlow_loss(codes_to_rows(a, idx), codes_to_rows(b, idx), lambd=0.3)[0]
    assert torch.autograd.gradcheck(f, (z1, z2))

def test_dead_channel():
    torch.manual_seed(0)
    z1 = torch.randn(2, 8, 16, 16, requires_grad=True)
    z2 = torch.randn(2, 8, 16, 16, requires_grad=True)
    mask = torch.ones(1, 8, 1, 1); mask[:,0] = 0
    loss, on, off, C = barlow_loss(codes_to_rows(z1*mask), codes_to_rows(z2*mask))
    loss.backward()
    assert torch.isfinite(loss) and C[0].abs().max() == 0
    assert torch.isfinite(z1.grad).all() and torch.isfinite(z2.grad).all()

def test_rows_alignment():
    torch.manual_seed(0)
    B, M, H, W = 3, 5, 4, 6
    z = torch.randn(B, M, H, W)
    rows = codes_to_rows(z)
    b, h, w = 2, 1, 3
    assert torch.equal(rows[b*H*W + h*W + w], z[b,:,h,w])
    idx = sample_pixels(z, 30)
    assert len(idx) == 30 and len(idx.unique()) == 30
    assert torch.equal(codes_to_rows(z, idx), rows[idx])
    assert sample_pixels(z, None) is None and sample_pixels(z, B*H*W) is None
    # identical views stay aligned under subsampling
    assert barlow_loss(codes_to_rows(z, idx), codes_to_rows(z.clone(), idx))[1] < 1e-6

def test_projector():
    torch.manual_seed(0)
    p = Projector(16, dims=(32, 32, 8))
    r = torch.randn(100, 16, requires_grad=True)
    out = p(r)
    assert out.shape == (100, 8)
    out.sum().backward()
    assert r.grad is not None and r.grad.abs().sum() > 0
    n_lin = sum(isinstance(l, torch.nn.Linear) for l in p.layers)
    n_bn  = sum(isinstance(l, torch.nn.BatchNorm1d) for l in p.layers)
    assert (n_lin, n_bn) == (3, 2)

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"PASS {name}")
    print(f"all {len(tests)} tests passed.")
