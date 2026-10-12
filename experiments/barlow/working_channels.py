#!/usr/bin/env python3
""" What do working channels carry? Working channels are active inside the unrolled iterations but
switched off in the final code (transient_off in inloop_masking.py). For each such channel j and
iteration k, the code map z_k[j] is regressed on candidate signals at code resolution:
    lowpass_y2, lowpass_y4   noisy image y, Gaussian-blurred (sigma 2 / 4 px)
    local_std                local standard deviation of y (sigma 2 px window)
    band_y, band_x, band_n   the channel's own analysis filter A_k[j] applied to y, clean x, noise n
Single-signal R^2 (squared correlation), plus a joint fit z = b_x band_x + b_n band_n + c:
b_n ~ 0 means the channel holds a clean band, b_n ~ b_x the raw noisy band.
Also |A_k^T r_k|[j] / |z_{k-1}[j]|: correction term vs carried state in z_k = ST(z_{k-1} - A_k^T r_k).
Random live channels (active in the final code) are the reference. Run from the repo root:
    python experiments/barlow/working_channels.py trained_nets/barlow/final
Writes results.json, carriers.png, example_maps.png to <runs_root>/working (or --out).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
from model.net import ST
from model.utils import pre_process
from explain_atoms import load, RC, INK, INK2, MUTED, GRID
from inloop_masking import channel_groups, RUNS

SIGNALS = ["lowpass_y2", "lowpass_y4", "local_std", "band_y", "band_x", "band_n"]
SIG_COLOR = {"lowpass_y2": "#2a78d6", "lowpass_y4": "#86b6ef", "local_std": "#1baf7a",
             "band_y": INK2, "band_x": "#eb6834", "band_n": "#e87ba4"}

def gauss_blur(t, s):
    r = int(3*s)
    g = torch.exp(-torch.arange(-r, r+1, dtype=t.dtype)**2 / (2*s**2)); g = g / g.sum()
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))

@torch.no_grad()
def trace(net, x, y, sigma, keep=None):
    """ codes z_k, correction terms A_k^T r_k, and candidate signals for one image.
    keep: channels to store (all if None); stored channel i is keep[i].
    """
    sel = (lambda t: t[:, keep]) if keep is not None else (lambda t: t)
    yp, params, _ = pre_process(y, net.s)
    pad = params[1]
    prep = lambda t: F.pad(t - t.mean(), pad, mode="reflect")
    xp, npad = prep(x), prep(y - x)
    down = lambda t: t[..., ::net.s, ::net.s]
    generic = {"lowpass_y2": down(gauss_blur(yp, 2)), "lowpass_y4": down(gauss_blur(yp, 4)),
               "local_std": down((gauss_blur(yp**2, 2) - gauss_blur(yp, 2)**2).clamp(min=0).sqrt())}
    c = sigma/255.0 if net.adaptive else 0
    z = ST(net.A[0].T(yp), net.t[0,:1] + c*net.t[0,1:2])
    zs, corrs = [sel(z)], [None]
    bands = [{"band_y": sel(net.A[0].T(yp)), "band_x": sel(net.A[0].T(xp)), "band_n": sel(net.A[0].T(npad))}]
    for k in range(1, net.K):
        r = net.B[k](z) - yp
        u = net.A[k].T(r)
        z = ST(z - u, net.t[k,:1] + c*net.t[k,1:2])
        zs.append(sel(z)); corrs.append(sel(u))
        bands.append({"band_y": sel(net.A[k].T(yp)), "band_x": sel(net.A[k].T(xp)), "band_n": sel(net.A[k].T(npad))})
    return zs, corrs, bands, generic

def r2(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = (a.norm()*b.norm()).item()
    return (a @ b).item()**2 / d**2 if d > 0 else 0.0

def joint(z, bx, bn):
    X = torch.stack([bx, bn, torch.ones_like(bx)], dim=1).double()
    sol = torch.linalg.lstsq(X, z.double()[:, None]).solution.flatten()
    res = z.double() - X @ sol
    tot = ((z - z.mean())**2).sum().item()
    return sol[0].item(), sol[1].item(), 1 - (res**2).sum().item() / tot if tot > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str)
    parser.add_argument("--runs", type=str, nargs="+", default=RUNS)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test", type=str, default="dataset/CBSD68/")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--n_images", type=int, default=6)
    parser.add_argument("--n_live", type=int, default=10, help="random live channels used as reference.")
    parser.add_argument("--crop", type=int, default=256, help="center crop of each test image (bounds memory).")
    parser.add_argument("--activity_cache", type=str, default=None)
    ARGS = parser.parse_args()
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "working")
    os.makedirs(out_dir, exist_ok=True)
    cache = torch.load(ARGS.activity_cache or os.path.join(ARGS.runs_root, "explain", "activity_cache.pt"))

    loader = data.get_data_loader([ARGS.test], test=True)
    def center(x, c=ARGS.crop):
        H, W = x.shape[-2:]
        return x[..., (H - min(c, H))//2:(H - min(c, H))//2 + min(c, H), (W - min(c, W))//2:(W - min(c, W))//2 + min(c, W)]
    images = [center(x) for _, x in zip(range(ARGS.n_images), loader)]
    g = torch.Generator().manual_seed(0)
    noisy = [x + ARGS.sigma/255*torch.randn(x.shape, generator=g) for x in images]
    rng = np.random.default_rng(0)

    results, example = {}, None
    for run in ARGS.runs:
        net = load(os.path.join(ARGS.runs_root, run))
        groups = channel_groups(cache[run])
        working = groups["transient_off"]
        live = sorted(rng.choice(groups["live"], min(ARGS.n_live, len(groups["live"])), replace=False).tolist())
        if not working:
            print(f"{run}: no working channels, skipped"); continue
        keep = working + live
        traces = [trace(net, x, y, ARGS.sigma, keep) for x, y in zip(images, noisy)]
        K = net.K
        def stats(j):
            j = keep.index(j)  # position among the stored channels
            out = {s: [] for s in SIGNALS}; out.update({"b_x": [], "b_n": [], "r2_joint": [], "update_ratio": [], "activity": []})
            for k in range(K):
                z = torch.cat([t[0][k][0, j].flatten() for t in traces])
                out["activity"].append((z != 0).float().mean().item())
                for s in SIGNALS[:3]:
                    out[s].append(r2(z, torch.cat([t[3][s][0, 0].flatten() for t in traces])))
                bx, bn, by = [torch.cat([t[2][k][b][0, j].flatten() for t in traces]) for b in ["band_x", "band_n", "band_y"]]
                out["band_x"].append(r2(z, bx)); out["band_n"].append(r2(z, bn)); out["band_y"].append(r2(z, by))
                b_x, b_n, rj = joint(z, bx, bn)
                out["b_x"].append(b_x); out["b_n"].append(b_n); out["r2_joint"].append(rj)
                if k == 0:
                    out["update_ratio"].append(np.nan)
                else:
                    u = torch.cat([t[1][k][0, j].flatten() for t in traces])
                    zp = torch.cat([t[0][k-1][0, j].flatten() for t in traces])
                    out["update_ratio"].append((u.norm() / zp.norm().clamp(min=1e-12)).item())
            return out
        per = {"working": {j: stats(j) for j in working}, "live": {j: stats(j) for j in live}}
        summ = {grp: {key: np.nanmedian([per[grp][j][key] for j in per[grp]], axis=0).tolist()
                      for key in SIGNALS + ["b_x", "b_n", "r2_joint", "update_ratio", "activity"]} for grp in per}
        results[run] = {"working_channels": working, "live_reference": live, "median": summ,
                        "per_channel": {grp: {str(j): v for j, v in per[grp].items()} for grp in per}}
        mid = slice(10, 29)
        w = summ["working"]
        print(f"{run}: {len(working)} working channels | iterations 10-28 median R^2: " +
              ", ".join(f"{s} {np.mean(w[s][mid]):.2f}" for s in SIGNALS) +
              f" | joint R^2 {np.mean(w['r2_joint'][mid]):.2f}, b_x {np.mean(w['b_x'][mid]):.2f}, b_n {np.mean(w['b_n'][mid]):.2f}"
              f" | update/state {np.mean(w['update_ratio'][mid]):.2f}")
        if example is None or len(working) > len(example[1]):
            example = (run, working, traces, keep)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"sigma": ARGS.sigma, "n_images": ARGS.n_images, "runs": results}, f)

    # ---- figure: per run, R^2 of each signal vs iteration (working vs live), joint gains, update ratio
    runs = list(results)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(len(runs), 4, figsize=(15, 2.9*len(runs)), squeeze=False, sharex=True)
        for i, run in enumerate(runs):
            m = results[run]["median"]; ks = np.arange(len(m["working"]["band_x"]))
            for col, grp in enumerate(["working", "live"]):
                ax = axs[i, col]
                for s in SIGNALS:
                    ax.plot(ks, m[grp][s], color=SIG_COLOR[s], lw=1.6, label=s)
                ax.set_ylim(0, 1); ax.grid(True, color=GRID, lw=0.6)
                ax.set_title(f"{run.replace('final_', '')}: {'working' if grp == 'working' else 'live (reference)'} channels",
                             fontsize=8.5)
                ax.set_ylabel("R² (median over channels)")
            ax = axs[i, 2]
            for grp, ls in [("working", "-"), ("live", "--")]:
                ax.plot(ks, m[grp]["b_x"], color=SIG_COLOR["band_x"], ls=ls, lw=1.6, label=f"b_x {grp}")
                ax.plot(ks, m[grp]["b_n"], color=SIG_COLOR["band_n"], ls=ls, lw=1.6, label=f"b_n {grp}")
            ax.axhline(0, color=MUTED, lw=0.8); ax.grid(True, color=GRID, lw=0.6)
            ax.set_title("joint fit z ≈ b_x·band_x + b_n·band_n", fontsize=8.5)
            ax = axs[i, 3]
            for grp, ls in [("working", "-"), ("live", "--")]:
                ax.plot(ks, m[grp]["update_ratio"], color=INK2, ls=ls, lw=1.6, label=grp)
                ax.plot(ks, m[grp]["activity"], color=MUTED, ls=ls, lw=1.2, label=f"activity {grp}")
            ax.set_yscale("log"); ax.grid(True, color=GRID, lw=0.6)
            ax.set_title("|A_kᵀr_k| / |z_{k−1}| per channel, and activity", fontsize=8.5)
        for ax in axs[-1]:
            ax.set_xlabel("iteration k")
        axs[0, 0].legend(fontsize=7, ncol=2); axs[0, 2].legend(fontsize=7); axs[0, 3].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "carriers.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---- figure: one working channel's code next to its candidate signals (run with most working channels)
    run, working, traces, keep = example
    zs, _, bands, generic = traces[0]
    j = max(working, key=lambda j: np.mean(results[run]["per_channel"]["working"][str(j)]["r2_joint"][10:29]))
    p, k = keep.index(j), 18
    maps = [(f"z_{k}[{j}] (working)", zs[k][0, p]), ("lowpass y (2 px)", generic["lowpass_y2"][0, 0]),
            ("local std of y", generic["local_std"][0, 0]), (f"A_{k}[{j}] ⋆ y", bands[k]["band_y"][0, p]),
            (f"A_{k}[{j}] ⋆ x (clean)", bands[k]["band_x"][0, p]), (f"A_{k}[{j}] ⋆ n (noise)", bands[k]["band_n"][0, p])]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, len(maps), figsize=(2.3*len(maps), 2.6))
        for ax, (t, m) in zip(axs, maps):
            m = m.numpy(); v = np.percentile(np.abs(m - np.median(m)), 99) or 1
            ax.imshow(m, cmap="gray", vmin=np.median(m) - v, vmax=np.median(m) + v, interpolation="nearest")
            ax.set_title(t, fontsize=8, color=INK); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{run.replace('final_', '')}, one test image, σ={ARGS.sigma:g}", fontsize=9, color=INK2, y=1.02)
        fig.subplots_adjust(wspace=0.05)
        fig.savefig(os.path.join(out_dir, "example_maps.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote results.json, carriers.png, example_maps.png to {out_dir}")

if __name__ == "__main__":
    main()
