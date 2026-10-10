#!/usr/bin/env python3
""" tests for GDLNet envelope clamping (a_max). run from repo root: python tests/test_net.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.net import GDLNet
from model.metrics import get_filters, effective_support

def all_a(net):
    return torch.cat([op.a.flatten() for k in range(net.K) for op in [net.A[k], net.B[k]]])

def test_no_clamp_unchanged():
    torch.manual_seed(0); net0 = GDLNet(K=3, M=16, P=11, s=2)
    torch.manual_seed(0); net1 = GDLNet(K=3, M=16, P=11, s=2, a_max=None)
    sd0, sd1 = net0.state_dict(), net1.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    assert all_a(net0).abs().max() > 1  # randn init exceeds any small bound
    with torch.no_grad():
        net0.A[1].a.fill_(5.0)
    net0.project()
    assert torch.all(net0.A[1].a == 5.0)

def test_clamp_init_and_project():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=16, P=11, s=2, a_max=0.625)
    assert all_a(net).abs().max() <= 0.625
    with torch.no_grad():
        net.A[2].a.fill_(5.0); net.B[1].a.fill_(-5.0); net.t.fill_(-1.0)
    net.project()
    assert torch.all(net.A[2].a == 0.625) and torch.all(net.B[1].a == -0.625)
    assert torch.all(net.t == 0)

def test_clamp_prevents_collapse():
    # |a| <= a_max => envelope energy std >= ~1/(2 a_max) px on the major axis (up to grid sampling).
    # Clamped randn init puts many atoms at the bound, so a_max needs margin below 1/(2 collapse_px).
    torch.manual_seed(0)
    free = GDLNet(K=2, M=169, P=11, s=2, init=False)
    assert (effective_support(get_filters(free.A[0]))[0] < 0.8).sum() > 20
    for a_max in [0.625, 0.5, 0.4]:
        torch.manual_seed(0)
        clamped = GDLNet(K=2, M=169, P=11, s=2, a_max=a_max, init=False)
        smaj = effective_support(get_filters(clamped.A[0]))[0]
        assert smaj.min() >= 1/(2*a_max) - 0.03, (a_max, smaj.min())
    assert (smaj < 0.8).sum() == 0  # a_max=0.4 -> >= 1.25px

def test_clamp_shared_params():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, shared="a_psi_w0_alpha", a_max=0.5)
    assert net.A[2].a is net.A[0].a
    with torch.no_grad():
        net.A[0].a.fill_(2.0)
    net.project()
    assert torch.all(net.A[2].a == 0.5)

def test_clamp_trains_and_reloads():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, a_max=0.5)
    opt = torch.optim.Adam(net.parameters(), lr=1.0)  # large steps push |a| past the bound
    x = torch.rand(2, 1, 32, 32)
    for _ in range(5):
        opt.zero_grad()
        xh, _ = net(x + 0.1*torch.randn_like(x), torch.full((2,1,1,1), 25.))
        torch.mean((xh - x)**2).backward()
        opt.step(); net.project()
        assert all_a(net).abs().max() <= 0.5
    net2 = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, a_max=0.5, init=False)
    net2.load_state_dict(net.state_dict())
    assert all(torch.equal(net.state_dict()[k], net2.state_dict()[k]) for k in net.state_dict())

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"PASS {name}")
    print(f"all {len(tests)} tests passed.")
