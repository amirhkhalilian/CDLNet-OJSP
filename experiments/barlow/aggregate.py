#!/usr/bin/env python3
""" Aggregate GDLNet + Barlow Twins runs: tables + figures.
Each immediate subdirectory of runs_root with an args.json is a run. Run from the repo root:
    python experiments/barlow/aggregate.py trained_nets/barlow/pilot
    python experiments/barlow/aggregate.py trained_nets/barlow
Reads per run: args.json, coherence/coherence.json (falls back to the last dict_metrics.jsonl
entry), test_<test_set>_None.txt (from analyze.py --test), val.txt, barlow.txt, backtrack.txt.
Writes to <runs_root>/aggregate (or --out): runs.csv, groups.csv, summary.md,
tradeoff.png, training.png, layers.png.
"""
import os, sys, json, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated reference palette (dataviz skill): categorical slots 1-2 + secondary ink, blue ordinal ramp
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAMILY_COLOR = {"ref": INK2, "noproj": "#2a78d6", "proj": "#eb6834"}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
VIEW_STYLE = {"single": ("s", ":"), "same": ("o", "-"), "diff": ("^", "--")}
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
      "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
      "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
      "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
      "axes.spines.top": False, "axes.spines.right": False,
      "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
      "legend.labelcolor": INK2, "font.size": 10}

# ----------------------------------------------------------------------------- loading

def condition(args):
    """ (cond, beta, view, family) of a run from its args.json.
    """
    bt = args["train"]["fit"].get("barlow")
    if bt is None or bt.get("views", 2) == 1:
        return "c0", None, "single", "ref"
    view = bt.get("view_sigma", "same")
    beta = float(bt.get("beta", 0.0))
    if beta == 0:
        return "c1", 0.0, view, "ref"
    if bt.get("projector") is not None:
        return ("c4" if view == "same" else "c5"), beta, view, "proj"
    return ("c2" if view == "same" else "c3"), beta, view, "noproj"

def read_floats(fn):
    if not os.path.exists(fn):
        return []
    return [float(v) for v in open(fn).read().replace("\n", ",").split(",") if v.strip()]

def read_test(fn):
    """ {sigma: psnr} from analyze.py test log (later lines win).
    """
    out = {}
    if os.path.exists(fn):
        for line in open(fn):
            if line.strip():
                s, p = line.split(",")
                out[float(s)] = float(p)
    return out

def read_dict_metrics(fn):
    """ {epoch: entry}, keeping the last entry per epoch (backtracks/resumes repeat epochs).
    """
    out = {}
    if os.path.exists(fn):
        for line in open(fn):
            if line.strip():
                m = json.loads(line)
                out[m["epoch"]] = m
    return dict(sorted(out.items()))

def load_run(run_dir, test_set):
    args = json.load(open(os.path.join(run_dir, "args.json")))
    cond, beta, view, family = condition(args)
    dm = read_dict_metrics(os.path.join(run_dir, "dict_metrics.jsonl"))
    coh_fn = os.path.join(run_dir, "coherence", "coherence.json")
    if os.path.exists(coh_fn):
        final, source = json.load(open(coh_fn)), "coherence"
    elif dm:
        final, source = list(dm.values())[-1], "dict_metrics"
    else:
        final, source = None, "none"

    r = {"run": os.path.basename(os.path.normpath(run_dir)), "cond": cond, "beta": beta,
         "view": view, "family": family, "seed": args.get("seed"), "metrics_source": source}
    if final is not None:
        r.update({"A_mu_s": final["A_mean"]["mu_s"], "A_n_dup": final["A_mean"]["n_dup_pairs"],
                  "A_mean_s": final["A_mean"]["mean_s"], "A_mu_0": final["A_mean"]["mu_0"],
                  "D_mu_s": final["D"]["mu_s"], "D_n_dup": final["D"]["n_dup_pairs"],
                  "layers_mu_s": [a["mu_s"] for a in final["A"]],
                  "layers_n_dup": [a["n_dup_pairs"] for a in final["A"]]})
    val = read_floats(os.path.join(run_dir, "val.txt"))
    r["val_psnr"] = val[-1] if val else np.nan
    for s, p in read_test(os.path.join(run_dir, f"test_{test_set}_None.txt")).items():
        r[f"psnr_{s:g}"] = p
    bt_fn = os.path.join(run_dir, "barlow.txt")
    if os.path.exists(bt_fn):
        rows = [l.split(",") for l in open(bt_fn).read().splitlines()[1:] if l.strip()]
        r["mean_Cii"] = float(rows[-1][5]) if rows else np.nan
    r["n_backtracks"] = len(open(os.path.join(run_dir, "backtrack.txt")).read().split()) \
        if os.path.exists(os.path.join(run_dir, "backtrack.txt")) else 0
    r["curve_epoch"] = list(dm.keys())
    r["curve_n_dup"] = [m["A_mean"]["n_dup_pairs"] for m in dm.values()]
    r["curve_mu_s"]  = [m["A_mean"]["mu_s"] for m in dm.values()]
    return r

# ----------------------------------------------------------------------------- grouping

SCALARS = ["A_mu_s", "A_n_dup", "A_mean_s", "A_mu_0", "D_mu_s", "D_n_dup", "val_psnr", "mean_Cii", "n_backtracks"]

def group_runs(runs):
    """ groups keyed by (cond, beta), seeds pooled. Curves are averaged over the epochs all seeds share.
    """
    groups = {}
    for r in runs:
        groups.setdefault((r["cond"], r["beta"]), []).append(r)
    out = []
    for (cond, beta), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        g = {"cond": cond, "beta": beta, "view": rs[0]["view"], "family": rs[0]["family"],
             "n": len(rs), "runs": rs, "seeds": sorted(r["seed"] for r in rs if r["seed"] is not None)}
        keys = SCALARS + sorted({k for r in rs for k in r if k.startswith("psnr_")}, key=lambda k: float(k[5:]))
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            g[k] = np.nanmean(v) if np.isfinite(v).any() else np.nan
            g[k + "_std"] = np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1 else np.nan
        common = sorted(set.intersection(*[set(r["curve_epoch"]) for r in rs]))
        g["curve_epoch"] = common
        for c in ["curve_n_dup", "curve_mu_s"]:
            g[c] = np.mean([[dict(zip(r["curve_epoch"], r[c]))[e] for e in common] for r in rs], axis=0) \
                if common else np.array([])
        for c in ["layers_mu_s", "layers_n_dup"]:
            g[c] = np.mean([r[c] for r in rs], axis=0) if all(c in r for r in rs) else None
        out.append(g)
    return out

def label(g, groups):
    """ condition name, with beta when that condition was swept.
    """
    n_beta = len({h["beta"] for h in groups if h["cond"] == g["cond"]})
    return g["cond"] if n_beta == 1 or g["beta"] is None else f"{g['cond']} β={g['beta']:g}"

# ----------------------------------------------------------------------------- tables

def fmt(g, k, digits=3):
    m, s = g.get(k, np.nan), g.get(k + "_std", np.nan)
    if not np.isfinite(m):
        return "-"
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if np.isfinite(s) else "")

def write_tables(runs, groups, out_dir, psnr_keys):
    run_cols = ["run", "cond", "beta", "view", "seed", "metrics_source"] + SCALARS + psnr_keys
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(run_cols)
        for r in runs:
            w.writerow([r.get(k, "") for k in run_cols])

    grp_cols = ["cond", "beta", "view", "n"] + [k + s for k in SCALARS + psnr_keys for s in ["", "_std"]]
    with open(os.path.join(out_dir, "groups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(grp_cols)
        for g in groups:
            w.writerow([g.get(k, "") for k in grp_cols])

    head = ["group", "n", "μ_s(A)", "n_dup(A)", "μ_0(A)", "μ_s(D)", "n_dup(D)", "val PSNR"] + \
           [f"PSNR σ={k[5:]}" for k in psnr_keys] + ["mean C_ii", "backtracks"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|"*len(head)]
    for g in groups:
        row = [label(g, groups), str(g["n"]), fmt(g, "A_mu_s"), fmt(g, "A_n_dup", 1), fmt(g, "A_mu_0"),
               fmt(g, "D_mu_s"), fmt(g, "D_n_dup", 1), fmt(g, "val_psnr", 2)] + \
              [fmt(g, k, 2) for k in psnr_keys] + [fmt(g, "mean_Cii"), fmt(g, "n_backtracks", 1)]
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"mean ± std over seeds (n runs per group). Metrics on analysis filters A are means over layers k.\n\n{table}\n")
    return table

# ----------------------------------------------------------------------------- figures

def legend_handles(groups):
    """ family (color) x view (marker/line) legend entries for the groups present.
    """
    from matplotlib.lines import Line2D
    names = {"ref": "no Barlow", "noproj": "Barlow", "proj": "Barlow + projector"}
    views = {"single": "1 view", "same": "2 views, same σ", "diff": "2 views, diff σ"}
    seen, hs = set(), []
    for g in groups:
        key = (g["family"], g["view"])
        if key in seen:
            continue
        seen.add(key)
        m, ls = VIEW_STYLE[g["view"]]
        hs.append(Line2D([], [], color=FAMILY_COLOR[g["family"]], marker=m, linestyle=ls,
                         label=f"{names[g['family']]}, {views[g['view']]} ({g['cond']})"))
    return hs

def plot_tradeoff(groups, out_dir, psnr_key):
    """ PSNR vs redundancy of A, one point per group (error bars over seeds), beta sweeps connected.
    """
    ylab = f"test PSNR, σ={psnr_key[5:]} (dB)" if psnr_key != "val_psnr" else "final val PSNR (dB)"
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, (xk, xlab) in zip(axs, [("A_mu_s", "μ_s(A), mean over k  (lower = more diverse)"),
                                        ("A_n_dup", "near-duplicate pairs in A, mean over k")]):
            for cond in sorted({g["cond"] for g in groups}):
                gs = sorted([g for g in groups if g["cond"] == cond], key=lambda g: g["beta"] or 0)
                c = FAMILY_COLOR[gs[0]["family"]]
                m, ls = VIEW_STYLE[gs[0]["view"]]
                x = np.array([g[xk] for g in gs]); y = np.array([g[psnr_key] for g in gs])
                ax.errorbar(x, y, xerr=[g[xk + "_std"] for g in gs], yerr=[g[psnr_key + "_std"] for g in gs],
                            color=c, marker=m, linestyle=ls if len(gs) > 1 else "none", linewidth=1.5,
                            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0, elinewidth=1)
                if len(gs) > 1: # label sweep endpoints only, the line gives the order
                    for g, xi, yi in [(gs[0], x[0], y[0]), (gs[-1], x[-1], y[-1])]:
                        ax.annotate(f"β={g['beta']:g}", (xi, yi), textcoords="offset points", xytext=(6, 4),
                                    fontsize=8, color=INK2)
            ax.set_xlabel(xlab)
        axs[0].set_ylabel(ylab)
        fig.subplots_adjust(bottom=0.3)
        fig.legend(handles=legend_handles(groups), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Denoising vs dictionary redundancy", color=INK)
        fn = os.path.join(out_dir, "tradeoff.png")
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fn

def plot_facets(groups, out_dir, fn_name, xkey, rows, xlabel, title):
    """ one column per Barlow condition (c2..c5), its beta(s) in the blue ramp, c0/c1 as gray references.
    rows: list of (ykey, ylabel).
    """
    refs  = [g for g in groups if g["family"] == "ref"]
    conds = sorted({g["cond"] for g in groups if g["family"] != "ref"}) or ["references"]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(len(rows), len(conds), figsize=(4.2*len(conds), 3.2*len(rows)),
                                sharex=True, sharey="row", squeeze=False)
        for j, cond in enumerate(conds):
            gs = sorted([g for g in groups if g["cond"] == cond], key=lambda g: g["beta"])
            ramp = BLUE_RAMP if len(gs) > 1 else [FAMILY_COLOR["noproj"]]
            idx  = np.linspace(0, len(ramp)-1, len(gs)).round().astype(int) if gs else []
            for i, (yk, ylab) in enumerate(rows):
                ax = axs[i, j]
                for g in refs:
                    x = g[xkey] if xkey != "layer" else np.arange(len(g[yk])) if g[yk] is not None else []
                    if g[yk] is None or len(g[yk]) == 0:
                        continue
                    m, ls = VIEW_STYLE[g["view"]]
                    ax.plot(x, g[yk], color=INK2, linestyle=ls, linewidth=1.5, label=label(g, groups))
                for g, k in zip(gs, idx):
                    if g[yk] is None or len(g[yk]) == 0:
                        continue
                    x = g[xkey] if xkey != "layer" else np.arange(len(g[yk]))
                    ax.plot(x, g[yk], color=ramp[k], label=f"β={g['beta']:g}" if len(gs) > 1 else "Barlow")
                if i == 0:
                    ax.set_title(cond if len(gs) != 1 else f"{cond} (β={gs[0]['beta']:g})")
                if j == 0:
                    ax.set_ylabel(ylab)
                if i == len(rows) - 1:
                    ax.set_xlabel(xlabel)
        # one legend for all panels, de-duplicated by label
        handles = {}
        for ax in axs.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
        fig.tight_layout()
        fig.suptitle(title, color=INK, y=1.02)
        fn = os.path.join(out_dir, fn_name)
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fn

# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=str, help="directory whose subdirectories are runs.")
    parser.add_argument("--out", type=str, default=None, help="output dir (default: <runs_root>/aggregate).")
    parser.add_argument("--test_set", type=str, default="CBSD68", help="test log name: test_<test_set>_None.txt.")
    parser.add_argument("--psnr_sigma", type=float, default=25, help="test sigma for the trade-off plot.")
    ARGS = parser.parse_args()

    run_dirs = sorted(os.path.join(ARGS.runs_root, d) for d in os.listdir(ARGS.runs_root)
                      if os.path.exists(os.path.join(ARGS.runs_root, d, "args.json")))
    if not run_dirs:
        sys.exit(f"no runs (subdirectories with args.json) in {ARGS.runs_root}")
    runs = [load_run(d, ARGS.test_set) for d in run_dirs]
    groups = group_runs(runs)
    out_dir = ARGS.out or os.path.join(ARGS.runs_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    psnr_keys = sorted({k for r in runs for k in r if k.startswith("psnr_")}, key=lambda k: float(k[5:]))
    psnr_key = f"psnr_{ARGS.psnr_sigma:g}"
    if psnr_key not in psnr_keys:
        print(f"no test PSNR at sigma={ARGS.psnr_sigma:g} ({ARGS.test_set}); trade-off plot uses final val PSNR.")
        psnr_key = "val_psnr"
    for g in groups:
        if len(g["seeds"]) != len(set(g["seeds"])):
            print(f"warning: group {label(g, groups)} pools runs with repeated seeds: {[r['run'] for r in g['runs']]}")
    missing = [r["run"] for r in runs if r["metrics_source"] != "coherence"]
    if missing:
        print(f"no coherence/coherence.json (using last dict_metrics.jsonl entry) for: {missing}")

    table = write_tables(runs, groups, out_dir, psnr_keys)
    figs = [plot_tradeoff(groups, out_dir, psnr_key),
            plot_facets(groups, out_dir, "training.png", "curve_epoch",
                        [("curve_n_dup", "near-dup pairs in A"), ("curve_mu_s", "μ_s(A)")],
                        "epoch", "Redundancy of A during training (mean over k)"),
            plot_facets(groups, out_dir, "layers.png", "layer",
                        [("layers_mu_s", "μ_s(A_k)"), ("layers_n_dup", "near-dup pairs in A_k")],
                        "k (iteration)", "Final redundancy per layer")]

    print(f"{len(runs)} runs in {len(groups)} groups from {ARGS.runs_root}\n")
    print(table)
    print(f"\nwrote runs.csv, groups.csv, summary.md, " + ", ".join(os.path.basename(f) for f in figs) + f" to {out_dir}")

if __name__ == "__main__":
    main()
