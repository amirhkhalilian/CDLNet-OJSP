#!/usr/bin/env python3
""" tests for GDLNet envelope clamping (a_max) and shape tying to D (tie_D).
run from repo root: python tests/test_net.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.net import GDLNet
from model.metrics import get_filters, effective_support, net_dict_metrics
from model.barlow import codes_to_rows, barlow_loss

SHAPE = ["a", "w0", "psi"]

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

def test_tie_D_shares_shape_not_alpha():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, tie_D=SHAPE)
    for k in range(3):
        for op in [net.A[k], net.B[k]]:
            assert all(getattr(op, n) is getattr(net.D, n) for n in SHAPE)
        assert net.A[k].alpha is not net.D.alpha
    assert len({id(op.alpha) for k in range(3) for op in [net.A[k], net.B[k]]}) == 6
    # parameter count: shared shape (5 per atom) + alpha per operator (2K) + thresholds (2 per layer)
    M, K = 8, 3
    assert sum(p.numel() for p in net.parameters()) == 5*M + 2*K*M + 2*K*M

def test_tie_D_filters_equal_D_up_to_scale():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, tie_D=SHAPE)
    unit = lambda W: W / W.flatten(1).norm(dim=1)[:,None,None,None]
    WD = unit(get_filters(net.D))
    for k in range(3):
        for op in [net.A[k], net.B[k]]:
            W = unit(get_filters(op))
            sign = torch.sign((W*WD).flatten(1).sum(1))[:,None,None,None]
            assert torch.allclose(W, sign*WD, atol=1e-5)
    m = net_dict_metrics(net)
    assert all(abs(m["A"][k][key] - m["D"][key]) < 1e-5 for k in range(3) for key in ["mu_s", "n_dup_pairs", "n_collapsed"])

def test_tie_D_barlow_reaches_D():
    # untied: barlow on codes sends no gradient to D; tied: D's shape gets gradient through the A_k
    for tie in [None, SHAPE]:
        torch.manual_seed(0)
        net = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, tie_D=tie)
        x = torch.rand(2, 1, 32, 32); s = torch.full((2,1,1,1), 25.)
        _, z1 = net(x + 0.1*torch.randn_like(x), s); _, z2 = net(x + 0.1*torch.randn_like(x), s)
        barlow_loss(codes_to_rows(z1), codes_to_rows(z2))[0].backward()
        g = sum(getattr(net.D, n).grad.abs().sum().item() for n in SHAPE if getattr(net.D, n).grad is not None)
        assert (g == 0) if tie is None else (g > 0), (tie, g)

def test_tie_D_trains_clamps_and_reloads():
    torch.manual_seed(0)
    net = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, tie_D=SHAPE, a_max=0.5)
    opt = torch.optim.Adam(net.parameters(), lr=1.0)
    x = torch.rand(2, 1, 32, 32)
    for _ in range(5):
        opt.zero_grad()
        xh, _ = net(x + 0.1*torch.randn_like(x), torch.full((2,1,1,1), 25.))
        torch.mean((xh - x)**2).backward()
        opt.step(); net.project()
    assert all(net.A[k].a is net.D.a and net.B[k].w0 is net.D.w0 for k in range(3))
    assert net.D.a.abs().max() <= 0.5
    sd = net.state_dict()
    net2 = GDLNet(K=3, M=8, P=7, s=2, adaptive=True, tie_D=SHAPE, a_max=0.5, init=False)
    net2.load_state_dict(sd)
    assert all(torch.equal(sd[k], net2.state_dict()[k]) for k in sd)
    assert all(net2.A[k].psi is net2.D.psi for k in range(3))  # still tied after loading

def test_tie_D_errors():
    for kw in [{"tie_D": ["alpha"]}, {"tie_D": ["w"]}, {"tie_D": SHAPE, "shared": "a_w0"}]:
        try:
            GDLNet(K=2, M=4, P=7, init=False, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")

def test_tie_D_via_init_model():
    # tie_D as given in an args.json model block (a JSON list)
    import train
    torch.manual_seed(0)
    args = {"type": "GDLNet", "model": {"K": 3, "M": 8, "P": 7, "s": 2, "tie_D": ["a", "w0", "psi"], "a_max": 0.5},
            "paths": {"save": os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_tie"), "ckpt": None},
            "train": {"opt": {"lr": 1e-3}, "sched": {"gamma": 0.95, "step_size": 50}}}
    try:
        net, *_ = train.init_model(args)
        assert net.A[0].a is net.D.a and net.tie_D == SHAPE
    finally:
        os.rmdir(args["paths"]["save"])

if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"PASS {name}")
    print(f"all {len(tests)} tests passed.")
