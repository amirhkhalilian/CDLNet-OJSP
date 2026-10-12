#!/usr/bin/env python3
""" E4: keep the statistics, destroy the content. Do transient channels (active inside the unrolled
iterations, off in the final code; transient_off in inloop_masking.py) help through their specific
content, or by supplying codes of the right size and statistics?

At every iteration, the transient channels' codes are overwritten with a substitute taken from an
unmodified run of the network; the rest of the network runs normally. Their unmodified final code is
~0, so every substitute keeps them out of the output as the network does, and only their role inside
the iterations is compared. The reference live channels are substituted at iterations 0..K-2 only, so
the substitute does not enter the output directly. Substitutes:
    own       the channels' own unmodified codes (sanity check: no change)
    swap      the same channels' codes for a different image (same iteration): realistic statistics,
              wrong content
    shift     the image's own codes moved by half the code size (circular): own statistics, misplaced
    sign      own magnitudes with a fixed random sign per coefficient: same support and size, content gone
    zero      masked, as in transient_role.py E2
The same treatments on an equal number of random live channels are the reference. Images of one size
are batched; the donor of image i is image i+1. Run from the repo root:
    python experiments/barlow/transient_swap.py trained_nets/barlow/final
Writes results.json, e4_dpsnr.png, e4_examples_outputs.png, e4_examples_changes.png and e4_codes.png
to <runs_root>/transient_swap (or --out).
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
from model.utils import pre_process, post_process
from explain_atoms import load, RC, INK, INK2, MUTED, GRID
from inloop_masking import channel_groups, RUNS
from transient_role import psnr, noise_split, radial_spectrum

MODES = ["own", "swap", "shift", "sign", "zero"]
LABEL = {"own": "own codes (sanity)", "swap": "codes of another image", "shift": "own codes, shifted",
         "sign": "own magnitudes, random signs", "zero": "masked (zero)"}
COL = {"transient": "#eb6834", "random_live": "#2a78d6"}

@torch.no_grad()
def substitute_loop(net, y, sigma, chans, mode, perm, sign=None, k_last=None, record=None):
    """ GDLNet forward on a batch where, at iterations 0..k_last, the codes of `chans` are replaced by a
    substitute built from the unmodified trajectory (run in lockstep). record: iteration k at which to
    also return the unmodified code (for figures). Returns xhat (and the unmodified code at `record`).
    """
    yp, params, _ = pre_process(y, net.s)
    c = sigma/255.0 if net.adaptive else 0
    k_last = net.K - 2 if k_last is None else k_last
    idx = list(chans)
    tau = lambda k: net.t[k,:1] + c*net.t[k,1:2]
    H, W = None, None

    def substitute(z, zb, k):
        if mode == "none" or k > k_last or not idx:
            return z
        z = z.clone()
        src = zb[:, idx]
        if mode == "own":
            z[:, idx] = src
        elif mode == "swap":
            z[:, idx] = src[perm]
        elif mode == "shift":
            z[:, idx] = torch.roll(src, shifts=(src.shape[-2]//2, src.shape[-1]//2), dims=(-2, -1))
        elif mode == "sign":
            z[:, idx] = src.abs() * sign
        elif mode == "zero":
            z[:, idx] = 0
        return z

    zb = ST(net.A[0].T(yp), tau(0))                  # unmodified trajectory
    z = substitute(zb, zb, 0)
    rec = zb if record == 0 else None
    for k in range(1, net.K):
        zb = ST(zb - net.A[k].T(net.B[k](zb) - yp), tau(k))
        z = substitute(ST(z - net.A[k].T(net.B[k](z) - yp), tau(k)), zb, k)
        if record == k:
            rec = zb
    xh = post_process(net.D(z), list(params))
    return (xh, rec) if record is not None else xh

def band_shares(base_err, err, n_bins=24):
    """ share of the added error power below 0.5 rad/px and above 2 rad/px (ring-area weighted). """
    b, f = radial_spectrum(base_err, n_bins); t, _ = radial_spectrum(err, n_bins)
    f = np.array(f); inc = np.clip(np.array(t) - np.array(b), 0, None)*f
    s = inc.sum() or 1
    return {"low_share": float(inc[f < 0.5].sum()/s), "high_share": float(inc[f >= 2.0].sum()/s)}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str)
    parser.add_argument("--runs", type=str, nargs="+", default=RUNS)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test", type=str, default="dataset/CBSD68/")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--n_images", type=int, default=12)
    parser.add_argument("--n_random", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--crop", type=int, default=192)
    parser.add_argument("--record_k", type=int, default=15, help="iteration shown in the code figure.")
    parser.add_argument("--activity_cache", type=str, default=None)
    ARGS = parser.parse_args()
    dev = torch.device(ARGS.device)
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "transient_swap")
    os.makedirs(out_dir, exist_ok=True)
    cache = torch.load(ARGS.activity_cache or os.path.join(ARGS.runs_root, "explain", "activity_cache.pt"))

    by_size = {}
    for x in data.get_data_loader([ARGS.test], test=True):
        by_size.setdefault(tuple(x.shape[-2:]), []).append(x)
    x = torch.cat(max(by_size.values(), key=len)[:ARGS.n_images]).to(dev)
    g = torch.Generator().manual_seed(0)
    n = (ARGS.sigma/255*torch.randn(x.shape, generator=g)).to(dev)
    y = x + n
    N = len(x); perm = torch.tensor([(i + 1) % N for i in range(N)], device=dev)
    rng = np.random.default_rng(0)
    H, W = x.shape[-2:]
    S = min(ARGS.crop, H, W); cy, cx = (H - S)//2, (W - S)//2
    crop = lambda t: t[ARGS.example, 0, cy:cy+S, cx:cx+S].detach().cpu()

    results, examples, code_fig = {}, {}, None
    for run in ARGS.runs:
        net = load(os.path.join(ARGS.runs_root, run)).to(dev)
        groups = channel_groups(cache[run])
        T, L = groups["transient_off"], groups["live"]
        if not T:
            print(f"{run}: no transient channels, skipped"); continue
        sets = {"transient": [T]} | {"random_live": [sorted(rng.choice(L, len(T), replace=False).tolist()) for _ in range(ARGS.n_random)]}
        code_shape = (N, len(T), (H + (H % net.s))//net.s, (W + (W % net.s))//net.s)
        sign = (torch.randint(0, 2, code_shape, generator=torch.Generator().manual_seed(1)).float()*2 - 1).to(dev)

        xh0 = substitute_loop(net, y, ARGS.sigma, [], "none", perm)
        base = psnr(x, xh0)
        res = {"n_transient": len(T), "base_psnr": base, "base": noise_split(x, xh0, n)}
        ex = {"clean": crop(x), "donor": crop(x[perm]), "base": crop(xh0)}
        for grp, chan_sets in sets.items():
            res[grp] = {}
            for mode in MODES:
                vals = []
                for i, chans in enumerate(chan_sets):
                    k_last = net.K - 1 if grp == "transient" else net.K - 2
                    xh = substitute_loop(net, y, ARGS.sigma, chans, mode, perm, sign=sign, k_last=k_last)
                    vals.append({"dpsnr": psnr(x, xh) - base, **noise_split(x, xh, n), **band_shares(xh0 - x, xh - x)})
                    if grp == "transient":
                        ex[mode] = crop(xh)
                res[grp][mode] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            print(f"{run} {grp:11s}: " + ", ".join(f"{m} {res[grp][m]['dpsnr']:+.2f}" for m in MODES) + " dB")
        results[run], examples[run] = res, ex

        # codes of one transient channel at iteration record_k (run with the most transient channels)
        if code_fig is None or len(T) > code_fig["n"]:
            _, zk = substitute_loop(net, y, ARGS.sigma, [], "none", perm, record=ARGS.record_k)
            act = (zk[:, T] != 0).float().mean(dim=(0, 2, 3))
            p = int(act.argmax()); j = T[p]
            own = zk[:, j]
            ch, cw = S//net.s, S//net.s; oy, ox = cy//net.s, cx//net.s
            cc = lambda t: t[ARGS.example, oy:oy+ch, ox:ox+cw].cpu()
            code_fig = {"n": len(T), "run": run, "j": j, "k": ARGS.record_k,
                        "maps": [("own code", cc(own)), ("code of another image", cc(own[perm])),
                                 ("own code, shifted", cc(torch.roll(own, (own.shape[-2]//2, own.shape[-1]//2), (-2, -1)))),
                                 ("own magnitudes, random signs", cc(own.abs()*sign[:, p]))],
                        "images": [("this image", crop(x)), ("the other image", crop(x[perm]))]}

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"sigma": ARGS.sigma, "n_images": N, "runs": results}, f, indent=1)
    plots(results, examples, code_fig, out_dir, ARGS.sigma)

def plots(results, examples, code_fig, out_dir, sigma):
    runs = list(results); short = lambda r: r.replace("final_", "").replace("_amax0.5", "")
    p = lambda a, b: -10*np.log10(torch.mean((a - b)**2).item())
    modes = MODES[1:]  # "own" is the sanity check; reported in results.json
    with plt.rc_context(RC):
        # PSNR change per substitute
        fig, axs = plt.subplots(1, len(runs), figsize=(3.4*len(runs), 3.2), sharey=True, squeeze=False)
        for ax, r in zip(axs[0], runs):
            for i, grp in enumerate(["transient", "random_live"]):
                v = [results[r][grp][m]["dpsnr"] for m in modes]
                ax.bar(np.arange(len(modes)) + (i - 0.5)*0.38, v, 0.38, color=COL[grp],
                       label="transient channels" if grp == "transient" else "random live channels (same number)")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(modes)), [LABEL[m].replace(", ", ",\n").replace("codes of another", "another\nimage's codes") for m in modes], fontsize=6.5)
            ax.set_title(f"{short(r)} ({results[r]['n_transient']} transient)", fontsize=8.5)
            ax.grid(True, axis="y", color=GRID, lw=0.6)
        axs[0][0].set_ylabel(f"PSNR change (dB), σ={sigma:g}"); axs[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "e4_dpsnr.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

        # example outputs and changes
        cols = [("clean", "clean x"), ("base", "network output x̂")] + [(m, f"transient: {LABEL[m]}") for m in modes]
        for name, show_change in [("e4_examples_outputs.png", False), ("e4_examples_changes.png", True)]:
            fig, axs = plt.subplots(len(runs), len(cols), figsize=(1.95*len(cols), 2.1*len(runs)), squeeze=False)
            for i, r in enumerate(runs):
                ex = examples[r]
                maps = {"clean": ex["clean"], "base": ex["base"] - ex["clean"], **{m: ex[m] - ex["base"] for m in modes}}
                v = max(np.percentile(maps[m].abs().numpy(), 99.5) for m in ["base"] + modes) or 1
                for j, (k, title) in enumerate(cols):
                    ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
                    if show_change and k != "clean":
                        ax.imshow(maps[k].numpy(), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                        sub = f"rms {maps[k].pow(2).mean().sqrt().item()*255:.1f}/255"
                        if k == "base":
                            title = "network error x̂ − x"
                    else:
                        ax.imshow(ex[k].clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        sub = "" if k == "clean" else f"{p(ex['clean'], ex[k]):.2f} dB"
                    ax.set_title(("\n".join(t for t in [title.replace(": ", ":\n", 1) if i == 0 else "", sub] if t)), fontsize=7, color=INK)
                axs[i, 0].set_ylabel(short(r), fontsize=8, color=INK2)
            fig.subplots_adjust(wspace=0.04, hspace=0.4)
            if show_change:
                fig.text(0.5, 0.0, "x̂ column: the network's own error; other columns: change from x̂ when the transient "
                         "channels' codes are replaced. Red = brighter, blue = darker; one color scale per row.",
                         ha="center", va="top", fontsize=8, color=INK2)
            fig.savefig(os.path.join(out_dir, name), dpi=170, bbox_inches="tight"); plt.close(fig)

        # what the substitutes look like for one transient channel
        cf = code_fig
        fig, axs = plt.subplots(1, 6, figsize=(13, 2.6))
        for ax, (t, im) in zip(axs[:2], cf["images"]):
            ax.imshow(im.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=8, color=INK)
        v = max(np.percentile(m.abs().numpy(), 99.5) for _, m in cf["maps"]) or 1
        for ax, (t, m) in zip(axs[2:], cf["maps"]):
            ax.imshow(m.numpy(), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_title(f"{t}\nactive {100*(m != 0).float().mean():.0f}%", fontsize=8, color=INK)
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{short(cf['run'])}: transient channel {cf['j']} at iteration {cf['k']} and its substitutes "
                     f"(code resolution, same crop)", fontsize=9, color=INK2, y=1.04)
        fig.savefig(os.path.join(out_dir, "e4_codes.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote results.json, e4_dpsnr.png, e4_examples_outputs.png, e4_examples_changes.png, e4_codes.png to {out_dir}")

if __name__ == "__main__":
    main()
