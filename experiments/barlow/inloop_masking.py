#!/usr/bin/env python3
""" In-loop masking of code channels in trained GDLNets: does the network need channels that are
active inside the iterations but switched off before synthesis (dead in the final code)?

Channel groups, from code activity on the test images (fraction of nonzero code coefficients):
    never      never active at any iteration
    transient_off  >= 99% active at some iteration, inactive (< 0.1%) in the final code. These channels
                   flicker: fully active in a few iterations, partly active in others (not dense throughout).
    live       active in the final code
A masked channel is zeroed after the soft-threshold of every iteration in a window, and the
network's remaining iterations run as usual. Random sets of live channels of the same size as
transient_off are the reference. Per iteration it records the relative residual |B_k z_{k-1} - y| / |y|
and the PSNR of the readout D z_k. Run from the repo root:
    python experiments/barlow/inloop_masking.py trained_nets/barlow/final
Writes results.json, delta_psnr.png, residual.png, readout_psnr.png to <runs_root>/inloop (or --out).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
from model.net import ST
from model.metrics import code_activity
from model.utils import pre_process, post_process
from explain_atoms import load, RC, INK2, MUTED, SURFACE

DEAD, PEAK = 1e-3, 0.99
RUNS = ["final_c1_amax0.5_tieD_seed0", "final_c1_amax0.5_tieD_seed1",
        "final_c2_amax0.5_tieD_seed0", "final_c2_amax0.5_tieD_seed1"]
COLOR = {"none": INK2, "transient_off": "#eb6834", "random_live": "#2a78d6", "never": MUTED}

def channel_groups(act):
    """ act: (K, M) code activity. """
    final_dead = act[-1] < DEAD
    never = act.max(dim=0).values < DEAD
    transient_off = final_dead & (act.max(dim=0).values >= PEAK)
    live = ~final_dead
    return {"never": torch.nonzero(never).flatten().tolist(),
            "transient_off": torch.nonzero(transient_off).flatten().tolist(),
            "live": torch.nonzero(live).flatten().tolist()}

@torch.no_grad()
def masked_forward(net, y, sigma, masked=(), window=None):
    """ GDLNet forward pass with channels `masked` zeroed after the soft-threshold of each
    iteration k in window (inclusive (k0, k1); None -> all). Returns xhat and per-iteration
    relative residual (k >= 1) and readout D z_k.
    """
    yp, params, _ = pre_process(y, net.s)
    c = sigma/255.0 if net.adaptive else 0
    keep = torch.ones(net.M, 1, 1)
    keep[list(masked)] = 0
    k0, k1 = window if window is not None else (0, net.K - 1)
    apply = lambda z, k: z*keep if k0 <= k <= k1 else z
    z = apply(ST(net.A[0].T(yp), net.t[0,:1] + c*net.t[0,1:2]), 0)
    resid, readout = [], [post_process(net.D(z), list(params))]
    for k in range(1, net.K):
        r = net.B[k](z) - yp
        resid.append((r.norm() / yp.norm()).item())
        z = apply(ST(z - net.A[k].T(r), net.t[k,:1] + c*net.t[k,1:2]), k)
        readout.append(post_process(net.D(z), list(params)))
    return readout[-1], resid, readout

def psnr(x, xh):
    return -10*np.log10(torch.mean((x - xh)**2).item())

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str)
    parser.add_argument("--runs", type=str, nargs="+", default=RUNS)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test", type=str, default="dataset/CBSD68/")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--n_images", type=int, default=12)
    parser.add_argument("--n_random", type=int, default=3, help="random live sets per window.")
    parser.add_argument("--windows", type=str, nargs="+", default=["0-28", "0-9", "10-19", "20-28"],
                        help="iteration windows k0-k1 (inclusive) in which channels are masked.")
    parser.add_argument("--activity_cache", type=str, default=None,
                        help="code-activity cache from explain_atoms.py (default: <runs_root>/explain/activity_cache.pt).")
    ARGS = parser.parse_args()
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "inloop")
    os.makedirs(out_dir, exist_ok=True)
    cache_fn = ARGS.activity_cache or os.path.join(ARGS.runs_root, "explain", "activity_cache.pt")
    cache = torch.load(cache_fn) if os.path.exists(cache_fn) else {}

    loader = data.get_data_loader([ARGS.test], test=True)
    images = [x for _, x in zip(range(ARGS.n_images), loader)]
    g = torch.Generator().manual_seed(0)
    noisy = [x + ARGS.sigma/255*torch.randn(x.shape, generator=g) for x in images]
    windows = [tuple(int(v) for v in w.split("-")) for w in ARGS.windows]
    rng = np.random.default_rng(0)

    results = {}
    for run in ARGS.runs:
        net = load(os.path.join(ARGS.runs_root, run))
        if run not in cache:
            print(f"code activity: {run} ...")
            cache[run] = code_activity(net, loader, ARGS.sigma)
        groups = channel_groups(cache[run])
        n_trans = len(groups["transient_off"])
        print(f"{run}: never {len(groups['never'])}, transient_off {n_trans}, live {len(groups['live'])}")

        def evaluate(masked, window):
            ps, rs, ro = [], [], []
            for x, y in zip(images, noisy):
                xh, resid, readout = masked_forward(net, y, ARGS.sigma, masked, window)
                ps.append(psnr(x, xh)); rs.append(resid); ro.append([psnr(x, r) for r in readout])
            return float(np.mean(ps)), np.mean(rs, axis=0).tolist(), np.mean(ro, axis=0).tolist()

        # sanity: the unmasked loop reproduces the network
        xh_net, _ = net(noisy[0], torch.full((1,1,1,1), ARGS.sigma))
        assert torch.allclose(masked_forward(net, noisy[0], ARGS.sigma)[0], xh_net, atol=1e-5)

        base_psnr, base_res, base_ro = evaluate((), None)
        res = {"groups": {k: len(v) for k, v in groups.items()}, "transient_off_channels": groups["transient_off"],
               "base": {"psnr": base_psnr, "residual": base_res, "readout_psnr": base_ro}, "masked": {}}
        for w in windows:
            key = f"{w[0]}-{w[1]}"
            entry = {}
            if n_trans:
                p, r, o = evaluate(groups["transient_off"], w)
                entry["transient_off"] = {"dpsnr": p - base_psnr, "residual": r, "readout_psnr": o}
                draws = [evaluate(rng.choice(groups["live"], n_trans, replace=False).tolist(), w)
                         for _ in range(ARGS.n_random)]
                entry["random_live"] = {"dpsnr": float(np.mean([d[0] for d in draws])) - base_psnr,
                                        "dpsnr_std": float(np.std([d[0] for d in draws])),
                                        "residual": np.mean([d[1] for d in draws], axis=0).tolist(),
                                        "readout_psnr": np.mean([d[2] for d in draws], axis=0).tolist()}
            if groups["never"]:
                p, r, o = evaluate(groups["never"], w)
                entry["never"] = {"dpsnr": p - base_psnr, "residual": r, "readout_psnr": o}
            res["masked"][key] = entry
            print(f"   window {key}: " + ", ".join(f"{g} {v['dpsnr']:+.3f} dB" for g, v in entry.items()))
        results[run] = res

    torch.save(cache, cache_fn)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"sigma": ARGS.sigma, "n_images": ARGS.n_images, "test": ARGS.test, "runs": results}, f, indent=1)

    # ---- figure 1: PSNR change per group and window
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, len(results), figsize=(3.6*len(results), 3.4), sharex=True, squeeze=False)
        for ax, (run, res) in zip(axs[0], results.items()):
            rows = []
            for key, entry in res["masked"].items():
                for grp in ["transient_off", "random_live", "never"]:
                    if grp in entry:
                        rows.append((f"{grp.replace('_', ' ')}, k {key}", entry[grp]["dpsnr"],
                                     entry[grp].get("dpsnr_std", 0), COLOR[grp]))
            for r, (lab, v, s, col) in enumerate(rows):
                ax.barh(r, v, xerr=s or None, color=col, height=0.62, error_kw={"elinewidth": 1, "ecolor": INK2})
                ax.text(v, r, f" {v:+.2f}" if v >= 0 else f"{v:+.2f} ", va="center", ha="left" if v >= 0 else "right",
                        fontsize=7, color=INK2)
            ax.axvline(0, color=MUTED, lw=1)
            ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
            ax.invert_yaxis()
            g = res["groups"]
            ax.set_title(f"{run.replace('final_', '')}\nbase {res['base']['psnr']:.2f} dB · {g['transient_off']} transient-off, "
                         f"{g['never']} never", fontsize=8.5)
            ax.set_xlabel(f"PSNR change (dB), σ={ARGS.sigma:g}")
        for ax in axs[0][1:]:
            ax.tick_params(labelleft=False)
        fig.subplots_adjust(wspace=0.06)
        fig.savefig(os.path.join(out_dir, "delta_psnr.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- figures 2-3: per-iteration residual and readout PSNR, masking over all iterations
    full = f"{windows[0][0]}-{windows[0][1]}"
    for name, key, ylab, logy in [("residual.png", "residual", "|B_k z_{k-1} − y| / |y|", True),
                                  ("readout_psnr.png", "readout_psnr", "PSNR of D z_k (dB)", False)]:
        with plt.rc_context(RC):
            fig, axs = plt.subplots(1, len(results), figsize=(3.6*len(results), 3.0), squeeze=False)
            for ax, (run, res) in zip(axs[0], results.items()):
                curves = [("no mask", res["base"][key], COLOR["none"], "-")]
                for grp in ["transient_off", "random_live", "never"]:
                    if grp in res["masked"][full]:
                        curves.append((f"{grp.replace('_', ' ')} masked", res["masked"][full][grp][key], COLOR[grp], "--" if grp == "never" else "-"))
                for lab, v, col, ls in curves:
                    k0 = 1 if key == "residual" else 0
                    ax.plot(range(k0, k0 + len(v)), v, color=col, linestyle=ls, lw=1.8, label=lab)
                if logy:
                    ax.set_yscale("log")
                ax.set_title(run.replace("final_", ""), fontsize=8.5)
                ax.set_xlabel("iteration k")
                ax.grid(True, color="#e1e0d9", lw=0.6)
            axs[0][0].set_ylabel(ylab)
            axs[0][0].legend(fontsize=7.5)
            fig.subplots_adjust(wspace=0.25)
            fig.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote results.json, delta_psnr.png, residual.png, readout_psnr.png to {out_dir}")

if __name__ == "__main__":
    main()
