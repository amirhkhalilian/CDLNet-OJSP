#!/usr/bin/env python3
""" Figures that explain the atom categories used in the dictionary metrics, on trained GDLNets:
distinct, duplicate (same atom, same position), near-duplicate (same atom up to a shift),
collapsed (spike/blob, no frequency selectivity) and dead (never used by the sparse code).
Run from the repo root on the final runs:
    python experiments/barlow/explain_atoms.py trained_nets/barlow/final
Writes PNG figures and numbers.json to <runs_root>/explain (or --out).
"""
import os, sys, json, argparse, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import train, data
from model.metrics import get_filters, xcorr, effective_support, code_activity, freq_order
from model.utils import pre_process, post_process, unpad

# thresholds of the metrics (model/metrics.py defaults)
THRESH, COLLAPSE_PX, DEAD = 0.9, 0.8, 1e-3
CATS = ["distinct", "duplicate", "collapsed", "dead"]
# dataviz reference palette: categorical slots 1-3 (validated all-pairs) + muted ink for "unused"
CAT_COLOR = {"distinct": "#2a78d6", "duplicate": "#eb6834", "collapsed": "#1baf7a", "dead": "#898781"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
      "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
      "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
      "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False}
ARMS = {"c0": "final_c0", "c1_clamp": "final_c1_amax0.5", "c2_clamp": "final_c2_amax0.5_lam0.05",
        "c1_clamp_tie": "final_c1_amax0.5_tieD", "c2_clamp_tie": "final_c2_amax0.5_tieD"}
ARM_LABEL = {"c0": "baseline (paper setting)", "c1_clamp": "envelope clamp", "c2_clamp": "clamp + Barlow (λ=0.05)",
             "c1_clamp_tie": "clamp + tie to D", "c2_clamp_tie": "clamp + tie + Barlow"}

# ----------------------------------------------------------------------------- model + categories

def load(run_dir):
    args = json.load(open(os.path.join(run_dir, "args.json")))
    with contextlib.redirect_stdout(io.StringIO()):
        net, *_ = train.init_model(args)
    return net.eval()

def categorize(W, act):
    """ per-atom category with priority dead > collapsed > duplicate > distinct, plus pair matrices.
    duplicate: shift-coherence > THRESH with another atom (includes zero-shift duplicates).
    """
    R  = xcorr(W).abs()
    P  = W.shape[-1]
    SC = R.amax(dim=(2,3)); Z = R[:,:,P-1,P-1]
    off = ~torch.eye(len(W), dtype=torch.bool)
    dup = ((SC > THRESH) & off).any(dim=1)
    collapsed = effective_support(W)[0] < COLLAPSE_PX
    dead = act < DEAD
    cat = np.array(["distinct"]*len(W), dtype=object)
    cat[dup.numpy()] = "duplicate"; cat[collapsed.numpy()] = "collapsed"; cat[dead.numpy()] = "dead"
    return cat, R, SC, Z

def best_shift(R, i, j):
    P = (R.shape[-1] + 1)//2
    k = R[i, j].flatten().argmax().item()
    return k // R.shape[-1] - (P-1), k % R.shape[-1] - (P-1)

def shift_atom(w, dy, dx):
    """ w shifted so that it aligns with the atom it was correlated against (zero fill). """
    P = w.shape[-1]
    out = torch.zeros_like(w)
    ys, xs = slice(max(dy,0), P+min(dy,0)), slice(max(dx,0), P+min(dx,0))
    yd, xd = slice(max(-dy,0), P+min(-dy,0)), slice(max(-dx,0), P+min(-dx,0))
    out[..., yd, xd] = w[..., ys, xs]
    return out

# ----------------------------------------------------------------------------- codes + reconstruction

@torch.no_grad()
def final_code(net, y, sigma):
    """ final sparse code z and the pre-processing params needed to synthesize from it. """
    s = torch.full((y.shape[0],1,1,1), float(sigma))
    *_, z, xhat = list(net.forward_generator(y, s))
    _, params, _ = pre_process(y, net.s)
    return z, params, xhat

@torch.no_grad()
def synth(net, z, params):
    return post_process(net.D(z), list(params))

@torch.no_grad()
def contribution(net, z, params, idx):
    """ part of the (mean-free) reconstruction that atoms idx are responsible for: D_idx z_idx """
    m = torch.zeros(z.shape[1], dtype=torch.bool); m[list(idx)] = True
    return unpad(net.D(z*m[None,:,None,None]), params[1])

def psnr(x, xh):
    return -10*np.log10(torch.mean((x - xh)**2).item())

def merge_duplicates(W, Z, act, z, cat):
    """ fold every zero-shift duplicate onto one representative per cluster: z_r += <d_r,d_j>/|d_r|^2 z_j.
    returns new code and the number of atoms freed.
    """
    M = len(W); live = act >= DEAD
    G = ((Z > THRESH) & ~torch.eye(M, dtype=torch.bool)).numpy() & live.numpy()[:,None] & live.numpy()[None,:]
    z, seen, freed = z.clone(), np.zeros(M, bool), 0
    Wf = W.flatten(1)
    for s in range(M):
        if seen[s] or not G[s].any():
            continue
        comp, stack = [], [s]; seen[s] = True
        while stack:
            v = stack.pop(); comp.append(v)
            for w in np.nonzero(G[v] & ~seen)[0]:
                seen[w] = True; stack.append(w)
        r = max(comp, key=lambda j: act[j].item())
        for j in comp:
            if j != r:
                z[:, r] += (Wf[r] @ Wf[j] / (Wf[r] @ Wf[r])) * z[:, j]
                z[:, j] = 0
                freed += 1
    return z, freed

# ----------------------------------------------------------------------------- drawing

def draw_atom(ax, w, color=None, lw=3):
    w = w[0].numpy() if torch.is_tensor(w) else w
    v = np.abs(w).max() or 1
    ax.imshow(w, cmap="gray", vmin=-v, vmax=v, interpolation="nearest")
    P = w.shape[-1]
    if color:
        ax.add_patch(Rectangle((-0.5,-0.5), P, P, fill=False, lw=lw, ec=color, clip_on=False))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def draw_fft(ax, w):
    X = np.abs(np.fft.fftshift(np.fft.fft2(w[0].numpy(), (64,64))))
    ax.imshow(X, cmap="Greys", vmin=0, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def draw_map(ax, m, title=None, sym=True):
    m = m.squeeze().numpy()
    v = np.abs(m).max() or 1
    ax.imshow(m, cmap="gray", vmin=-v if sym else 0, vmax=v, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, color=INK2)

def save(fig, out_dir, name):
    fn = os.path.join(out_dir, name)
    fig.savefig(fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fn}")

def atom_strip(out_dir, name, W, idx, color, notes, fft=True):
    """ one row of atoms (+ their frequency response below), each with a note. """
    n = len(idx)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2 if fft else 1, n, figsize=(1.25*n, 2.7 if fft else 1.5), squeeze=False)
        for c, (j, note) in enumerate(zip(idx, notes)):
            draw_atom(axs[0, c], W[j], color)
            axs[0, c].set_title(note, fontsize=7.5, color=INK2)
            if fft:
                draw_fft(axs[1, c], W[j])
        if fft:
            axs[1, 0].set_ylabel("|frequency\nresponse|", fontsize=7.5, color=INK2)
            axs[1, 0].yaxis.set_visible(True); axs[1, 0].set_yticks([])
        fig.subplots_adjust(wspace=0.15, hspace=0.25)
        save(fig, out_dir, name)

def pair_strip(out_dir, name, W, R, SC, Z, pairs, color, aligned=False):
    """ pairs of atoms side by side; with aligned, also the second atom shifted onto the first. """
    cols = 3 if aligned else 2
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, cols*len(pairs), figsize=(1.15*cols*len(pairs), 1.75), squeeze=False)
        for p, (i, j) in enumerate(pairs):
            dy, dx = best_shift(R, i, j)
            sign = 1 if (xcorr(W[[i, j]])[0, 1].flatten()[(R[i, j].flatten().argmax())] >= 0) else -1
            draw_atom(axs[0, cols*p], W[i], color); axs[0, cols*p].set_title(f"atom {i}", fontsize=7.5, color=INK2)
            draw_atom(axs[0, cols*p+1], W[j], color); axs[0, cols*p+1].set_title(f"atom {j}", fontsize=7.5, color=INK2)
            if aligned:
                draw_atom(axs[0, cols*p+2], sign*shift_atom(W[j], -dy, -dx), MUTED, lw=1.5)
                axs[0, cols*p+2].set_title(f"{j} moved ({-dy:+d},{-dx:+d})", fontsize=7.5, color=INK2)
            axs[0, cols*p].text(0, 1.32 if aligned else 1.3, f"shift-coh. {SC[i,j]:.2f} · zero-shift {Z[i,j]:.2f}",
                                transform=axs[0, cols*p].transAxes, fontsize=7.5, color=INK, ha="left")
        fig.subplots_adjust(wspace=0.12)
        save(fig, out_dir, name)

def dictionary_map(out_dir, name, W, cat, title):
    """ all atoms of a dictionary, grouped by category (then frequency), framed in the category color. """
    order = []
    for c in CATS:
        idx = np.nonzero(cat == c)[0]
        if len(idx):
            order += list(idx[freq_order(W[idx])])
    n = int(np.ceil(np.sqrt(len(W))))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(n, n, figsize=(0.42*n, 0.42*n + 0.5))
        for a in axs.flat:
            a.axis("off")
        for p, j in enumerate(order):
            ax = axs.flat[p]; ax.axis("on")
            draw_atom(ax, W[j], CAT_COLOR[cat[j]], lw=2.2)
        counts = ", ".join(f"{(cat == c).sum()} {c}" for c in CATS)
        fig.suptitle(f"{title}\n{counts}", fontsize=9, color=INK, y=1.0)
        fig.subplots_adjust(wspace=0.18, hspace=0.18)
        save(fig, out_dir, name)

# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--test", type=str, default="dataset/CBSD68/")
    parser.add_argument("--sigma", type=float, default=25)
    parser.add_argument("--n_ablation", type=int, default=12, help="test images used for the ablations.")
    parser.add_argument("--example_image", type=int, default=5, help="index of the test image for contribution maps.")
    ARGS = parser.parse_args()
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "explain")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    loader = data.get_data_loader([ARGS.test], test=True)
    images = [x for x in loader]
    numbers = {"thresholds": {"shift_coherence": THRESH, "collapse_px": COLLAPSE_PX, "dead_activity": DEAD,
                              "sigma": ARGS.sigma, "test": ARGS.test}}

    # ---- category counts for every final run (A: mean over k, D: final dictionary)
    cache_fn = os.path.join(out_dir, "activity_cache.pt")
    cache = torch.load(cache_fn) if os.path.exists(cache_fn) else {}
    comp, nets, acts = {}, {}, {}
    for arm, prefix in ARMS.items():
        for seed in [0, 1]:
            run = f"{prefix}_seed{seed}"
            net = load(os.path.join(ARGS.runs_root, run))
            if run not in cache:
                print(f"code activity: {run} ...")
                cache[run] = code_activity(net, loader, ARGS.sigma)
                torch.save(cache, cache_fn)
            act = cache[run]
            cats_A = [categorize(get_filters(net.A[k]), act[k])[0] for k in range(net.K)]
            cat_D = categorize(get_filters(net.D), act[-1])[0]
            comp[run] = {"A": {c: float(np.mean([(ck == c).sum() for ck in cats_A])) for c in CATS},
                         "D": {c: int((cat_D == c).sum()) for c in CATS}}
            if seed == 0:
                nets[arm], acts[arm] = net, act
    numbers["composition"] = comp

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
        arms = list(ARMS)
        for ax, bank in zip(axs, ["A", "D"]):
            for r, arm in enumerate(arms):
                vals = [np.mean([comp[f"{ARMS[arm]}_seed{s}"][bank][c] for s in [0, 1]]) for c in CATS]
                left = 0
                for c, v in zip(CATS, vals):
                    ax.barh(r, v, left=left, color=CAT_COLOR[c], height=0.62, edgecolor=SURFACE, linewidth=1.5)
                    if v >= 9:
                        ax.text(left + v/2, r, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
                    left += v
            ax.set_yticks(range(len(arms)), [ARM_LABEL[a] for a in arms])
            ax.invert_yaxis()
            ax.set_xlim(0, 169); ax.set_xlabel("atoms (of 169), mean of 2 seeds")
            ax.set_title("analysis filters A_k (mean over 30 layers)" if bank == "A" else "dictionary D")
        handles = [Rectangle((0,0), 1, 1, color=CAT_COLOR[c]) for c in CATS]
        fig.legend(handles, ["distinct (used, no near-copy)", "has a (near-)duplicate", "collapsed", "dead (unused)"],
                   loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
        fig.subplots_adjust(bottom=0.24, wspace=0.08)
        save(fig, out_dir, "composition.png")

    # ---- dictionaries of three arms, atoms framed by category
    for arm in ["c0", "c1_clamp", "c2_clamp"]:
        W = get_filters(nets[arm].D)
        cat, *_ = categorize(W, acts[arm][-1])
        dictionary_map(out_dir, f"dictionary_{arm}.png", W, cat, f"D, {ARM_LABEL[arm]} (seed 0)")

    # ---- galleries. collapsed atoms from the baseline; the rest from the clamped model (clean Gabors)
    W0 = get_filters(nets["c0"].D); a0 = acts["c0"][-1]
    cat0, R0, SC0, Z0 = categorize(W0, a0)
    W1 = get_filters(nets["c1_clamp"].D); a1 = acts["c1_clamp"][-1]
    cat1, R1, SC1, Z1 = categorize(W1, a1)
    smaj0 = effective_support(W0)[0]; smaj1 = effective_support(W1)[0]
    by_act = lambda idx, act: sorted(idx, key=lambda j: -act[j].item())

    distinct = by_act(np.nonzero(cat1 == "distinct")[0], a1)
    # spread the examples over frequency so they show different scales and orientations
    fo = [j for j in freq_order(W1) if j in set(distinct) and a1[j] >= 0.05]
    distinct_ex = [fo[int(i)] for i in np.linspace(0, len(fo)-1, 7).round()]
    atom_strip(out_dir, "gallery_distinct.png", W1, distinct_ex, CAT_COLOR["distinct"],
               [f"used {100*a1[j]:.0f}%" for j in distinct_ex])

    collapsed = by_act([j for j in np.nonzero(cat0 == "collapsed")[0]], a0)
    collapsed_ex = collapsed[:7]
    atom_strip(out_dir, "gallery_collapsed.png", W0, collapsed_ex, CAT_COLOR["collapsed"],
               [f"σ≈{smaj0[j]:.2f}px" for j in collapsed_ex])

    off = ~torch.eye(len(W1), dtype=torch.bool); live1 = (a1 >= DEAD) & (smaj1 >= COLLAPSE_PX)
    pairs = lambda mask: [(i, j) for i, j in torch.nonzero(torch.triu(mask & off & live1[:,None] & live1[None,:])).tolist()]
    zero_dups = sorted(pairs((Z1 > THRESH)), key=lambda p: -Z1[p].item())
    shifted   = sorted(pairs((SC1 > THRESH) & (Z1 < 0.5)), key=lambda p: -SC1[p].item())
    near_miss = sorted(pairs((SC1 > 0.75) & (SC1 < 0.85)), key=lambda p: -SC1[p].item())
    pick = lambda ps, n: [ps[int(i)] for i in np.linspace(0, len(ps)-1, min(n, len(ps))).round()] if ps else []
    pair_strip(out_dir, "gallery_duplicate.png", W1, R1, SC1, Z1, pick(zero_dups, 3), CAT_COLOR["duplicate"])
    pair_strip(out_dir, "gallery_near_duplicate.png", W1, R1, SC1, Z1, pick(shifted, 3), CAT_COLOR["duplicate"], aligned=True)
    pair_strip(out_dir, "gallery_below_threshold.png", W1, R1, SC1, Z1, pick(near_miss, 3), CAT_COLOR["distinct"], aligned=True)
    numbers["pairs_clamped_D"] = {"zero_shift_duplicates": len(zero_dups), "shifted_near_duplicates": len(shifted),
                                  "similar_below_threshold": len(near_miss)}

    # ---- dead atoms: they look like any other atom; their code is empty
    dead = [j for j in np.nonzero(cat1 == "dead")[0]][:4]
    live_ref = distinct[0]
    x = images[ARGS.example_image]
    g = torch.Generator().manual_seed(0)
    y = x + ARGS.sigma/255*torch.randn(x.shape, generator=g)
    z1, p1, _ = final_code(nets["c1_clamp"], y, ARGS.sigma)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 1 + len(dead), figsize=(1.6*(1+len(dead)), 3.3))
        for c, j in enumerate([live_ref] + dead):
            col = CAT_COLOR["distinct"] if c == 0 else CAT_COLOR["dead"]
            draw_atom(axs[0, c], W1[j], col)
            axs[0, c].set_title(f"atom {j}: used {100*a1[j]:.1f}%", fontsize=7.5, color=INK2)
            zj = z1[0, j].abs()
            vmax = max(np.percentile(z1[0, live_ref].abs().numpy(), 99), 1e-8)  # same scale for all panels
            axs[1, c].imshow(zj.numpy(), cmap="Greys", vmin=0, vmax=vmax, interpolation="nearest")
            axs[1, c].set_xticks([]); axs[1, c].set_yticks([])
            for s in axs[1, c].spines.values():
                s.set_color(AXIS)
        axs[1, 0].set_ylabel("|code| on a\ntest image", fontsize=7.5, color=INK2)
        fig.subplots_adjust(wspace=0.12, hspace=0.12)
        save(fig, out_dir, "gallery_dead.png")

    # ---- what each kind contributes to a reconstruction (baseline model has all four kinds)
    net = nets["c0"]
    z0, p0, xh0 = final_code(net, y, ARGS.sigma)
    assert torch.allclose(synth(net, z0, p0), xh0, atol=1e-5)
    live0 = (a0 >= DEAD).numpy()
    ex_distinct = by_act(np.nonzero(cat0 == "distinct")[0], a0)[0]
    ex_collapsed = collapsed[0]
    dup_pairs0 = [(i, j) for i, j in torch.nonzero(torch.triu((Z0 > THRESH) & ~torch.eye(len(W0), dtype=torch.bool))).tolist()
                  if live0[i] and live0[j] and smaj0[i] >= COLLAPSE_PX and smaj0[j] >= COLLAPSE_PX]
    di, dj = max(dup_pairs0, key=lambda p: min(a0[p[0]], a0[p[1]]).item()) if dup_pairs0 else (None, None)
    ex_dead = next(j for j in np.nonzero(cat0 == "dead")[0])
    H, Wd = x.shape[-2:]
    cy, cx = H//2 - 64, Wd//2 - 64
    crop = lambda t: t[..., cy:cy+128, cx:cx+128]
    panels = [("clean x", crop(x), False), ("noisy y (σ=25)", crop(y), False), ("denoised x̂", crop(xh0).clamp(0, 1), False),
              (f"distinct atom {ex_distinct}", crop(contribution(net, z0, p0, [ex_distinct])), True),
              (f"collapsed atom {ex_collapsed}", crop(contribution(net, z0, p0, [ex_collapsed])), True)]
    if di is not None:
        panels += [(f"duplicate atom {di}", crop(contribution(net, z0, p0, [di])), True),
                   (f"duplicate atom {dj}", crop(contribution(net, z0, p0, [dj])), True)]
    panels += [(f"dead atom {ex_dead}", crop(contribution(net, z0, p0, [ex_dead])), True)]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, len(panels), figsize=(1.75*len(panels), 2.1))
        for ax, (t, m, sym) in zip(axs, panels):
            if sym:
                draw_map(ax, m, t)
            else:
                ax.imshow(m.squeeze().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8, color=INK2)
                for s in ax.spines.values():
                    s.set_visible(False)
        fig.subplots_adjust(wspace=0.06)
        save(fig, out_dir, "contributions.png")

    # group sums vs the noise that was added
    n = (y - x)
    groups = {c: np.nonzero((cat0 == c) & live0)[0].tolist() for c in ["distinct", "collapsed", "duplicate"]}
    corr = {}
    for c, idx in groups.items():
        cg = contribution(net, z0, p0, idx)
        corr[c] = {"noise_corr": float((cg*n).sum() / (cg.norm()*n.norm())),
                   "signal_corr": float((cg*(x - x.mean())).sum() / (cg.norm()*(x - x.mean()).norm())),
                   "energy_share": float(cg.norm()**2 / unpad(net.D(z0), p0[1]).norm()**2)}
    numbers["baseline_group_contributions"] = corr
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.2))
        draw_map(axs[0], crop(n), "added noise y − x")
        draw_map(axs[1], crop(contribution(net, z0, p0, groups["collapsed"])),
                 f"all collapsed atoms\ncorr. with noise {corr['collapsed']['noise_corr']:.2f}")
        draw_map(axs[2], crop(contribution(net, z0, p0, groups["distinct"])),
                 f"all distinct atoms\ncorr. with noise {corr['distinct']['noise_corr']:.2f}")
        fig.subplots_adjust(wspace=0.06)
        save(fig, out_dir, "collapsed_vs_noise.png")

    # ---- ablations on the final code (no retraining): how much the trained model relies on each kind
    abl = {}
    for arm in ["c0", "c1_clamp", "c2_clamp"]:
        net, act = nets[arm], acts[arm][-1]
        W = get_filters(net.D); cat, R, SC, Z = categorize(W, act)
        rng = np.random.default_rng(0)
        rows = {k: [] for k in ["base", "drop_dead", "merge_dup", "drop_dup_copies", "drop_collapsed", "drop_random_distinct"]}
        freed = []
        for x in images[:ARGS.n_ablation]:
            y = x + ARGS.sigma/255*torch.randn(x.shape, generator=g)
            z, p, xh = final_code(net, y, ARGS.sigma)
            base = psnr(x, xh); rows["base"].append(base)
            keep = lambda idx: z * torch.tensor([j not in set(idx) for j in range(len(W))])[None,:,None,None]
            rows["drop_dead"].append(psnr(x, synth(net, keep(np.nonzero(cat == "dead")[0].tolist()), p)) - base)
            zm, nf = merge_duplicates(W, Z, act, z, cat); freed.append(nf)
            rows["merge_dup"].append(psnr(x, synth(net, zm, p)) - base)
            copies = [j for j in range(len(W)) if zm[:, j].abs().sum() == 0 and z[:, j].abs().sum() > 0]
            rows["drop_dup_copies"].append(psnr(x, synth(net, keep(copies), p)) - base)
            col = np.nonzero(cat == "collapsed")[0].tolist()
            rows["drop_collapsed"].append(psnr(x, synth(net, keep(col), p)) - base if col else np.nan)
            dist = np.nonzero(cat == "distinct")[0]
            k = max(len(col), len(copies), 1)
            rows["drop_random_distinct"].append(np.mean([psnr(x, synth(net, keep(rng.choice(dist, min(k, len(dist)), replace=False).tolist()), p)) - base
                                                         for _ in range(5)]))
        abl[arm] = {k: float(np.nanmean(v)) if not all(np.isnan(v)) else None for k, v in rows.items()}
        abl[arm]["n_dead"] = int((cat == "dead").sum()); abl[arm]["n_collapsed"] = int((cat == "collapsed").sum())
        abl[arm]["n_merged"] = float(np.mean(freed))
        abl[arm]["n_random"] = int(max((cat == "collapsed").sum(), np.mean(freed), 1))
    numbers["ablation_dB"] = abl

    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8), sharex=True)
        items = [("drop_dead", "remove dead atoms", "dead"), ("merge_dup", "merge duplicates onto one copy", "duplicate"),
                 ("drop_dup_copies", "remove duplicate copies (no merge)", "duplicate"),
                 ("drop_collapsed", "remove collapsed atoms", "collapsed"),
                 ("drop_random_distinct", "remove as many random distinct atoms", "distinct")]
        lo = min(v for a in abl.values() for k, v in a.items() if k in dict((i[0], 0) for i in items) and v is not None)
        for ax, arm in zip(axs, ["c0", "c1_clamp", "c2_clamp"]):
            for r, (k, lab, c) in enumerate(items):
                v = abl[arm][k]
                if v is None:
                    ax.text(0, r, "none to remove ", va="center", ha="right", fontsize=7.5, color=MUTED)
                    continue
                ax.barh(r, v, color=CAT_COLOR[c], height=0.6)
                txt = "0.00 dB" if abs(v) < 0.005 else f"{v:+.2f} dB"
                ax.text(v, r, f" {txt} " if v >= 0 or abs(v) < 0.005 else f"{txt} ", va="center",
                        ha="left" if v >= 0 or abs(v) < 0.005 else "right", fontsize=7.5, color=INK)
            ax.axvline(0, color=AXIS, lw=1)
            ax.set_yticks(range(len(items)), [i[1] for i in items])
            ax.invert_yaxis()
            ax.set_xlim(lo*1.45, -lo*0.35)
            ax.set_title(f"{ARM_LABEL[arm]}  (base {abl[arm]['base']:.2f} dB)")
            ax.set_xlabel("PSNR change (dB), σ=25")
        for ax in axs[1:]:
            ax.tick_params(labelleft=False)
        fig.subplots_adjust(wspace=0.08)
        save(fig, out_dir, "ablation.png")

    with open(os.path.join(out_dir, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    print(json.dumps({k: numbers[k] for k in ["pairs_clamped_D", "baseline_group_contributions", "ablation_dB"]}, indent=1))

if __name__ == "__main__":
    main()
