#!/usr/bin/env python3
""" How do transient channels (active inside the unrolled iterations, switched off in the final code;
transient_off in inloop_masking.py) help the reconstruction? Two experiments on trained GDLNets:

E1  let them in: lower their final-iteration threshold so D synthesizes them, and ask whether
    the content they add is noise (correlated with n = y - x) or missing signal (correlated with
    x - xhat). Reference: switch the same number of random live channels off at the end.
E2  mask them inside all iterations (as in inloop_masking.py) and decompose what goes wrong:
    error split into noise leakage (part of xhat - x along n) and the rest; radial error spectrum;
    and how the live code changes (support overlap, coefficient size, iteration-to-iteration
    oscillation). Reference: masking the same number of random live channels.

Images of one size are batched. Run from the repo root:
    python experiments/barlow/transient_role.py trained_nets/barlow/final
Writes results.json, e1_letin.png, e2_error.png, e2_spectrum.png, e2_codes.png, and example crops of
every condition (examples_outputs.png, examples_changes.png) to <runs_root>/transient (or --out).
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
from model.metrics import get_filters
from model.utils import pre_process, post_process
from explain_atoms import load, RC, INK, INK2, MUTED, GRID
from inloop_masking import channel_groups, RUNS

COL = {"base": INK2, "transient": "#eb6834", "random_live": "#2a78d6"}

# ----------------------------------------------------------------------------- the unrolled loop

@torch.no_grad()
def run_loop(net, y, sigma, masked=(), tau_last=None, track=()):
    """ GDLNet forward with channels `masked` zeroed after every soft-threshold, an optional
    final-iteration threshold override tau_last = (channels, value), and the per-iteration relative
    change |z_k - z_{k-1}| / |z_k| of the channels in `track`. Returns xhat, final code, changes.
    """
    yp, params, _ = pre_process(y, net.s)
    c = sigma/255.0 if net.adaptive else 0
    keep = torch.ones(1, net.M, 1, 1, device=y.device)
    keep[0, list(masked)] = 0
    track = list(track)
    tau = lambda k: net.t[k,:1] + c*net.t[k,1:2]
    z = ST(net.A[0].T(yp), tau(0)) * keep
    change = []
    for k in range(1, net.K):
        t = tau(k)
        if k == net.K - 1 and tau_last is not None:
            t = t.clone(); t[0, list(tau_last[0])] = tau_last[1]
        z_new = ST(z - net.A[k].T(net.B[k](z) - yp), t) * keep
        if track:
            d, n = (z_new - z)[:, track], z_new[:, track]
            change.append((d.norm() / n.norm().clamp(min=1e-12)).item())
        z = z_new
    return post_process(net.D(z), list(params)), z, change

# ----------------------------------------------------------------------------- measurements

def psnr(x, xh):
    return -10*np.log10(torch.mean((x - xh)**2).item())

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return ((a*b).sum() / (a.norm()*b.norm()).clamp(min=1e-12)).item()

def noise_split(x, xh, n):
    """ error e = xhat - x split into its component along the noise n and the rest (per-pixel MSE). """
    e = (xh - x).flatten(1); nn = n.flatten(1)
    beta = (e*nn).sum(1, keepdim=True) / (nn*nn).sum(1, keepdim=True)
    leak = beta*nn
    return {"mse": e.pow(2).mean().item(), "noise_leak": leak.pow(2).mean().item(),
            "other": (e - leak).pow(2).mean().item(), "beta": beta.mean().item()}

def radial_spectrum(e, n_bins=24):
    """ radially averaged power spectrum of a batch of images, bins over [0, pi] rad/px. """
    E = torch.fft.fft2(e.double().cpu()).abs().pow(2).mean(dim=(0, 1))
    H, W = E.shape
    fy = torch.fft.fftfreq(H)[:, None]*2*np.pi; fx = torch.fft.fftfreq(W)[None, :]*2*np.pi
    r = torch.sqrt(fy**2 + fx**2)
    edges = torch.linspace(0, np.pi, n_bins + 1)
    return [E[(r >= edges[i]) & (r < edges[i+1])].mean().item() for i in range(n_bins)], \
           ((edges[:-1] + edges[1:])/2).tolist()

def atom_spectrum(W, n_bins=24, n=64):
    X = torch.fft.fft2(W.double().cpu(), s=(n, n)).abs().pow(2).sum(dim=1).mean(dim=0)
    f = torch.fft.fftfreq(n)*2*np.pi
    r = torch.sqrt(f[:, None]**2 + f[None, :]**2)
    edges = torch.linspace(0, np.pi, n_bins + 1)
    return [X[(r >= edges[i]) & (r < edges[i+1])].mean().item() for i in range(n_bins)]

def code_compare(z_ref, z, channels):
    """ support overlap (Jaccard) and coefficient-size ratio of the code on `channels`, vs z_ref. """
    a, b = z_ref[:, channels], z[:, channels]
    sa, sb = a != 0, b != 0
    both = sa & sb
    jac = (both.sum() / (sa | sb).sum().clamp(min=1)).item()
    size = (b[both].abs().mean() / a[both].abs().mean().clamp(min=1e-12)).item() if both.any() else float("nan")
    return {"jaccard": jac, "size_ratio": size, "active_ref": sa.float().mean().item(), "active": sb.float().mean().item()}

# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str)
    parser.add_argument("--runs", type=str, nargs="+", default=RUNS)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test", type=str, default="dataset/CBSD68/")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--n_images", type=int, default=12, help="test images (of the most common size, batched).")
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu, mps or cuda. On an M-series Mac this model ran ~2x faster on cpu than on mps.")
    parser.add_argument("--activity_cache", type=str, default=None)
    parser.add_argument("--example", type=int, default=0, help="batch index of the image shown in the example figures.")
    parser.add_argument("--crop", type=int, default=192, help="side of the center crop shown in the example figures.")
    ARGS = parser.parse_args()
    dev = torch.device(ARGS.device)
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "transient")
    os.makedirs(out_dir, exist_ok=True)
    cache = torch.load(ARGS.activity_cache or os.path.join(ARGS.runs_root, "explain", "activity_cache.pt"))

    by_size = {}
    for x in data.get_data_loader([ARGS.test], test=True):
        by_size.setdefault(tuple(x.shape[-2:]), []).append(x)
    x = torch.cat(max(by_size.values(), key=len)[:ARGS.n_images]).to(dev)
    g = torch.Generator().manual_seed(0)
    n = (ARGS.sigma/255*torch.randn(x.shape, generator=g)).to(dev)
    y = x + n
    rng = np.random.default_rng(0)
    H, W_ = x.shape[-2:]
    S = min(ARGS.crop, H, W_); cy, cx = (H - S)//2, (W_ - S)//2
    crop = lambda t: t[ARGS.example, 0, cy:cy+S, cx:cx+S].detach().cpu()

    results, examples = {}, {}
    for run in ARGS.runs:
        net = load(os.path.join(ARGS.runs_root, run)).to(dev)
        groups = channel_groups(cache[run])
        T, L = groups["transient_off"], groups["live"]
        if not T:
            print(f"{run}: no transient channels, skipped"); continue
        draws = [sorted(rng.choice(L, len(T), replace=False).tolist()) for _ in range(ARGS.n_random)]
        tau_last = (net.t[-1, 0] + ARGS.sigma/255*net.t[-1, 1])[:, 0, 0]
        tau_live = tau_last[L].median().item()
        res = {"n_transient": len(T), "transient": T, "tau_last_transient": tau_last[T].median().item(), "tau_last_live": tau_live}

        xh0, z0, ch0 = run_loop(net, y, ARGS.sigma, track=L)
        ex = {"clean": crop(x), "noisy": crop(y), "base": crop(xh0)}
        res["base"] = {"psnr": psnr(x, xh0), **noise_split(x, xh0, n), "change": ch0}
        spec0, freqs = radial_spectrum(xh0 - x)
        res["base"]["spectrum"] = spec0

        # ---- E1: let the transient channels into the output; reference: switch random live channels off at the end
        e1 = {}
        for name, val in [("let_in_live_tau", tau_live), ("let_in_zero_tau", 0.0)]:
            xh, _, _ = run_loop(net, y, ARGS.sigma, tau_last=(T, val))
            ex[name] = crop(xh)
            added = xh - xh0
            e1[name] = {"dpsnr": psnr(x, xh) - res["base"]["psnr"], "corr_with_noise": corr(added, n),
                        "corr_with_missing_signal": corr(added, x - xh0),
                        "added_energy_rel": (added.norm() / (xh0 - xh0.mean()).norm()).item()}
        offs = []
        for D_ in draws:
            xh, _, _ = run_loop(net, y, ARGS.sigma, tau_last=(D_, 1e6))
            ex.setdefault("random_live_off_at_end", crop(xh))  # first draw
            removed = xh0 - xh
            offs.append({"dpsnr": psnr(x, xh) - res["base"]["psnr"], "corr_with_noise": corr(removed, n),
                         "corr_with_missing_signal": corr(removed, x - xh), "added_energy_rel": (removed.norm() / (xh0 - xh0.mean()).norm()).item()})
        e1["random_live_off_at_end"] = {k: float(np.mean([o[k] for o in offs])) for k in offs[0]}
        res["E1"] = e1

        # ---- E2: mask inside all iterations
        e2 = {}
        xh, z, ch = run_loop(net, y, ARGS.sigma, masked=T, track=L)
        ex["transient_masked"] = crop(xh)
        e2["transient"] = {"psnr": psnr(x, xh), **noise_split(x, xh, n), **code_compare(z0, z, L),
                           "change": ch, "spectrum": radial_spectrum(xh - x)[0]}
        reps = []
        for D_ in draws:
            others = [j for j in L if j not in set(D_)]
            xh, z, ch = run_loop(net, y, ARGS.sigma, masked=D_, track=others)
            ex.setdefault("random_live_masked", crop(xh))  # first draw
            ch_ref = run_loop(net, y, ARGS.sigma, track=others)[2]
            reps.append({"psnr": psnr(x, xh), **noise_split(x, xh, n), **code_compare(z0, z, others),
                         "change": ch, "change_ref": ch_ref, "spectrum": radial_spectrum(xh - x)[0]})
        e2["random_live"] = {k: (np.mean([r[k] for r in reps], axis=0).tolist() if isinstance(reps[0][k], list)
                                 else float(np.mean([r[k] for r in reps]))) for k in reps[0]}
        res["E2"] = e2
        W = get_filters(net.D).cpu()
        res["atom_spectrum"] = {"transient": atom_spectrum(W[T]), "live": atom_spectrum(W[L])}
        res["freqs"] = freqs
        results[run], examples[run] = res, ex

        b, t, r = res["base"], e2["transient"], e2["random_live"]
        print(f"{run}: {len(T)} transient | final τ transient {res['tau_last_transient']:.3f} vs live {tau_live:.3f}")
        for k, v in e1.items():
            print(f"   E1 {k:24s} ΔPSNR {v['dpsnr']:+.3f} dB | content corr. with noise {v['corr_with_noise']:+.2f}, "
                  f"with missing signal {v['corr_with_missing_signal']:+.2f} | energy {v['added_energy_rel']:.3f}")
        for name, v in [("base", b), ("mask transient", t), ("mask random live", r)]:
            extra = f" | live code: support overlap {v['jaccard']:.2f}, size ×{v['size_ratio']:.2f}, active {100*v['active']:.1f}% (ref {100*v['active_ref']:.1f}%)" if "jaccard" in v else ""
            print(f"   E2 {name:17s} PSNR {v['psnr']:.2f} | MSE {v['mse']:.2e} = noise leak {v['noise_leak']:.2e} + other {v['other']:.2e}{extra}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"sigma": ARGS.sigma, "n_images": ARGS.n_images, "device": ARGS.device, "runs": results}, f, indent=1)
    plots(results, out_dir, ARGS.sigma)
    plot_examples(examples, out_dir, ARGS.sigma)

EXAMPLE_COLS = [("base", "network output x̂"), ("let_in_live_tau", "E1: transient let in\n(live τ)"),
                ("let_in_zero_tau", "E1: transient let in\n(τ = 0)"), ("random_live_off_at_end", "E1 ref: random live\noff at end"),
                ("transient_masked", "E2: transient masked\nin all iterations"), ("random_live_masked", "E2 ref: random live\nmasked in all iterations")]

def plot_examples(examples, out_dir, sigma):
    """ example crops: outputs of every condition, and each condition's change from the network output. """
    runs = list(examples); short = lambda r: r.replace("final_", "").replace("_amax0.5", "")
    p = lambda a, b: -10*np.log10(torch.mean((a - b)**2).item())
    with plt.rc_context(RC):
        # outputs
        cols = [("clean", "clean x"), ("noisy", f"noisy y (σ={sigma:g})")] + EXAMPLE_COLS
        fig, axs = plt.subplots(len(runs), len(cols), figsize=(1.9*len(cols), 2.05*len(runs)), squeeze=False)
        for i, r in enumerate(runs):
            ex = examples[r]
            for j, (k, title) in enumerate(cols):
                ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(ex[k].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                lab = title if i == 0 else ""
                if k != "clean":
                    lab = (lab + "\n" if lab else "") + f"{p(ex['clean'], ex[k]):.2f} dB"
                ax.set_title(lab, fontsize=7.5, color=INK)
            axs[i, 0].set_ylabel(short(r), fontsize=8, color=INK2)
        fig.subplots_adjust(wspace=0.04, hspace=0.3)
        fig.savefig(os.path.join(out_dir, "examples_outputs.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

        # changes vs the network output (first panel: the network's own error), one scale per row
        cols = [("err", "network error x̂ − x")] + [(k, t.replace("network output x̂", "")) for k, t in EXAMPLE_COLS[1:]]
        fig, axs = plt.subplots(len(runs), len(cols), figsize=(1.9*len(cols), 2.05*len(runs)), squeeze=False)
        for i, r in enumerate(runs):
            ex = examples[r]
            maps = {"err": ex["base"] - ex["clean"], **{k: ex[k] - ex["base"] for k, _ in EXAMPLE_COLS[1:]}}
            v = max(np.percentile(m.abs().numpy(), 99.5) for m in maps.values()) or 1
            for j, (k, title) in enumerate(cols):
                ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(maps[k].numpy(), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                lab = (title + "\n" if i == 0 else "") + (f"rms {maps[k].pow(2).mean().sqrt().item()*255:.1f}/255")
                ax.set_title(lab, fontsize=7.5, color=INK)
            axs[i, 0].set_ylabel(f"{short(r)}\n±{v*255:.0f}/255", fontsize=8, color=INK2)
        fig.subplots_adjust(wspace=0.04, hspace=0.35)
        fig.text(0.5, 0.0, "first panel: the network's own error x̂ − x; other panels: each condition's change from x̂. "
                 "Red = brighter, blue = darker; one color scale per row.", ha="center", va="top", fontsize=8, color=INK2)
        fig.savefig(os.path.join(out_dir, "examples_changes.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote examples_outputs.png, examples_changes.png to {out_dir}")

def plots(results, out_dir, sigma):
    runs = list(results); short = lambda r: r.replace("final_", "").replace("_amax0.5", "")
    with plt.rc_context(RC):
        # E1
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
        items = [("let_in_live_tau", "transient let in (live τ)", COL["transient"]),
                 ("let_in_zero_tau", "transient let in (τ = 0)", "#f3a37f"),
                 ("random_live_off_at_end", "random live switched off at end", COL["random_live"])]
        w = 0.26
        for col, (key, ylab) in enumerate([("dpsnr", "PSNR change (dB)"), ("corr_with_noise", "corr. of added content with noise n"),
                                           ("corr_with_missing_signal", "corr. with missing signal x − x̂")]):
            ax = axs[col]
            for i, (k, lab, c) in enumerate(items):
                ax.bar(np.arange(len(runs)) + (i-1)*w, [results[r]["E1"][k][key] for r in runs], w, color=c, label=lab)
            ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(range(len(runs)), [short(r) for r in runs], fontsize=7, rotation=15)
            ax.set_ylabel(ylab); ax.grid(True, axis="y", color=GRID, lw=0.6)
        axs[0].legend(fontsize=7)
        fig.suptitle(f"E1: what the transient channels would add to the output (σ={sigma:g})", color=INK, fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "e1_letin.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

        # E2 error decomposition
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for i, r in enumerate(runs):
            for j, (name, v) in enumerate([("base", results[r]["base"]), ("transient", results[r]["E2"]["transient"]),
                                           ("random_live", results[r]["E2"]["random_live"])]):
                xpos = i*4 + j
                ax.bar(xpos, v["other"], color=COL[name], width=0.8)
                ax.bar(xpos, v["noise_leak"], bottom=v["other"], color=COL[name], alpha=0.45, width=0.8, hatch="///", edgecolor="white")
        ax.set_xticks([i*4 + 1 for i in range(len(runs))], [short(r) for r in runs], fontsize=7.5)
        ax.set_ylabel("output MSE"); ax.grid(True, axis="y", color=GRID, lw=0.6)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=COL["base"], label="no mask"), Patch(color=COL["transient"], label="transient masked"),
                           Patch(color=COL["random_live"], label="random live masked"),
                           Patch(facecolor="white", edgecolor=INK2, hatch="///", label="hatched: noise leakage (along n)")], fontsize=7)
        ax.set_title("E2: output error split into noise leakage and the rest", fontsize=10)
        fig.savefig(os.path.join(out_dir, "e2_error.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

        # E2 spectrum
        fig, axs = plt.subplots(1, len(runs), figsize=(3.6*len(runs), 3.0), squeeze=False)
        for ax, r in zip(axs[0], runs):
            f = results[r]["freqs"]; b = np.array(results[r]["base"]["spectrum"])
            ax.plot(f, np.array(results[r]["E2"]["transient"]["spectrum"]) / b, color=COL["transient"], lw=1.8, label="transient masked / base")
            ax.plot(f, np.array(results[r]["E2"]["random_live"]["spectrum"]) / b, color=COL["random_live"], lw=1.8, label="random live masked / base")
            ax2 = ax.twinx()
            for grp, c in [("transient", COL["transient"]), ("live", COL["random_live"])]:
                s = np.array(results[r]["atom_spectrum"][grp]); ax2.plot(f, s / s.max(), color=c, lw=1, ls=":")
            ax2.set_yticks([]); ax.axhline(1, color=MUTED, lw=0.8)
            ax.set_title(short(r), fontsize=8.5); ax.set_xlabel("radial frequency (rad/px)"); ax.grid(True, color=GRID, lw=0.6)
        axs[0][0].set_ylabel("error power ratio"); axs[0][0].legend(fontsize=7)
        fig.suptitle("E2: where the error grows (solid), and the atoms' frequency content (dotted, scaled)", fontsize=10, color=INK)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "e2_spectrum.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

        # E2 live code change per iteration
        fig, axs = plt.subplots(1, len(runs), figsize=(3.6*len(runs), 3.0), squeeze=False)
        for ax, r in zip(axs[0], runs):
            k = np.arange(1, 1 + len(results[r]["base"]["change"]))
            ax.plot(k, results[r]["base"]["change"], color=COL["base"], lw=1.6, label="no mask")
            ax.plot(k, results[r]["E2"]["transient"]["change"], color=COL["transient"], lw=1.6, label="transient masked")
            ax.plot(k, results[r]["E2"]["random_live"]["change"], color=COL["random_live"], lw=1.6, label="random live masked")
            ax.set_yscale("log"); ax.grid(True, color=GRID, lw=0.6)
            e = results[r]["E2"]
            ax.set_title(f"{short(r)}\nlive support overlap: transient-masked {e['transient']['jaccard']:.2f}, random {e['random_live']['jaccard']:.2f}", fontsize=8)
            ax.set_xlabel("iteration k")
        axs[0][0].set_ylabel("|z_k − z_{k−1}| / |z_k|, live channels"); axs[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "e2_codes.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote results.json, e1_letin.png, e2_error.png, e2_spectrum.png, e2_codes.png to {out_dir}")

if __name__ == "__main__":
    main()
