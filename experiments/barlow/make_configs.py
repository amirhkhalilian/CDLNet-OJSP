#!/usr/bin/env python3
""" Generate train.py arg files for the GDLNet + Barlow Twins experiments.
See experiments/barlow/README.md. Run from the repo root:
    python experiments/barlow/make_configs.py --smoke
    python experiments/barlow/make_configs.py --pilot
    python experiments/barlow/make_configs.py --pilot2
    python experiments/barlow/make_configs.py --beta B --beta_proj BP
"""
import os, json, argparse

CONDITIONS  = ["c0", "c1", "c2", "c3", "c4", "c5"]
LAMBDA0     = 5e-3                     # off-diagonal weight at D = M, scaled as LAMBDA0*M/D
PROJ_DIMS   = [2048, 2048, 2048]
N_PIXELS    = 16384                    # code pixels sampled per batch (projector conditions)
PILOT_BETAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
PILOT_EPOCHS = 300
SEEDS       = [0, 1]
# pilot 2: Barlow (c1, c2) x envelope clamp x shape tying to D, + off-diagonal weight sweep (clamped)
PILOT2_BETA    = 1e-3
PILOT2_A_MAX   = 0.5
PILOT2_TIE_D   = ["a", "w0", "psi"]
PILOT2_LAMBDAS = [0.05, 0.5]

def base_args(name, seed, save_root, data_root, epochs=6000):
    """ GDLNet paper settings (GDLNet-S, MoG order 1, untied), sigma in [10,40].
    """
    return {"type": "GDLNet",
            "seed": seed,
            "model": {"K": 30, "M": 169, "P": 11, "s": 2, "C": 1,
                      "order": 1, "adaptive": True},
            "paths": {"save": os.path.join(save_root, name), "ckpt": None},
            "train": {
                "loaders": {"batch_size": 10,
                            "crop_size": 128,
                            "trn_path_list": [os.path.join(data_root, "CBSD432")],
                            "val_path_list": [os.path.join(data_root, "Kodak")],
                            "tst_path_list": [os.path.join(data_root, "CBSD68")],
                            "load_color": False},
                "fit": {"epochs": epochs,
                        "noise_std": [10, 40],
                        "val_freq": 50,
                        "save_freq": 5,
                        "backtrack_thresh": 2,
                        "verbose": True,
                        "clip_grad": 5e-2,
                        "demosaic": False,
                        "mcsure": False},
                "opt": {"lr": 1e-3},
                "sched": {"gamma": 0.95, "step_size": 50}}}

def barlow_block(cond, beta, M, proj_dims=PROJ_DIMS, n_pixels=N_PIXELS):
    """ train.fit.barlow args for condition cond (None for c0).
    c1: two views (same sigma), beta=0 | c2/c3: same/diff sigma | c4/c5: c2/c3 + projector
    """
    if cond == "c0":
        return None
    proj = cond in ["c4", "c5"]
    D = proj_dims[-1] if proj else M
    bt = {"views":      2,
          "view_sigma": "diff" if cond in ["c3", "c5"] else "same",
          "beta":       0.0 if cond == "c1" else beta,
          "lambda":     LAMBDA0*M/D}
    if proj:
        bt["projector"] = {"dims": list(proj_dims)}
        bt["n_pixels"]  = n_pixels
    return bt

def make(name, cond, beta, seed, save_root, data_root, epochs=6000, smoke=False):
    args = base_args(name, seed, save_root, data_root, epochs)
    proj_dims, n_pixels = PROJ_DIMS, N_PIXELS
    if smoke:
        # tiny model and data so every condition runs end-to-end on a CPU in seconds
        args["model"].update({"K": 3, "M": 8})
        args["train"]["loaders"].update({"batch_size": 4, "crop_size": 64,
            "trn_path_list": [os.path.join(data_root, "Set12")],
            "val_path_list": [os.path.join(data_root, "Set12")],
            "tst_path_list": [os.path.join(data_root, "Set12")]})
        args["train"]["fit"].update({"val_freq": 1, "save_freq": 1})
        proj_dims, n_pixels = [64, 64, 32], 256
    bt = barlow_block(cond, beta, args["model"]["M"], proj_dims, n_pixels)
    if bt is not None:
        args["train"]["fit"]["barlow"] = bt
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="tiny c0-c5 configs for a local end-to-end check.")
    mode.add_argument("--pilot", action="store_true", help=f"beta sweep {PILOT_BETAS} on c2, c4 (+ c0, c1 references), {PILOT_EPOCHS} epochs, seed 0.")
    mode.add_argument("--pilot2", action="store_true", help=f"c1, c2 (beta={PILOT2_BETA:g}) x a_max {{None, {PILOT2_A_MAX:g}}} x "
                      f"tie_D {{None, {PILOT2_TIE_D}}}, + clamped c2 at lambda {PILOT2_LAMBDAS}; {PILOT_EPOCHS} epochs, seed 0.")
    parser.add_argument("--beta", type=float, help="beta* for c2, c3 (full mode).")
    parser.add_argument("--beta_proj", type=float, help="beta* for c4, c5 (full mode).")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="seeds per condition (full mode).")
    parser.add_argument("--data_root", type=str, default="dataset", help="directory containing CBSD432, Kodak, CBSD68, Set12.")
    parser.add_argument("--save_root", type=str, default=None, help="where runs are saved (default depends on mode).")
    ARGS = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    configs = {}
    if ARGS.smoke:
        out_dir   = os.path.join(here, "configs", "smoke")
        save_root = ARGS.save_root or "experiments/barlow/smoke_runs"
        for c in CONDITIONS:
            name = f"smoke_{c}"
            configs[name] = make(name, c, 1e-3, 0, save_root, ARGS.data_root, epochs=2, smoke=True)
    elif ARGS.pilot:
        out_dir   = os.path.join(here, "configs", "pilot")
        save_root = ARGS.save_root or "trained_nets/barlow/pilot"
        for c in ["c0", "c1"]:
            name = f"pilot_{c}"
            configs[name] = make(name, c, 0.0, 0, save_root, ARGS.data_root, epochs=PILOT_EPOCHS)
        for c in ["c2", "c4"]:
            for beta in PILOT_BETAS:
                name = f"pilot_{c}_beta{beta:g}"
                configs[name] = make(name, c, beta, 0, save_root, ARGS.data_root, epochs=PILOT_EPOCHS)
    elif ARGS.pilot2:
        out_dir   = os.path.join(here, "configs", "pilot2")
        save_root = ARGS.save_root or "trained_nets/barlow/pilot2"
        variants  = [(c, a_max, tie, None) for c in ["c1", "c2"] for a_max in [None, PILOT2_A_MAX] for tie in [False, True]] + \
                    [("c2", PILOT2_A_MAX, tie, lambd) for tie in [False, True] for lambd in PILOT2_LAMBDAS]
        for c, a_max, tie, lambd in variants:
            name = f"pilot2_{c}" + (f"_amax{a_max:g}" if a_max else "") + ("_tieD" if tie else "") + \
                   (f"_lam{lambd:g}" if lambd else "")
            args = make(name, c, PILOT2_BETA, 0, save_root, ARGS.data_root, epochs=PILOT_EPOCHS)
            if a_max:
                args["model"]["a_max"] = a_max
            if tie:
                args["model"]["tie_D"] = list(PILOT2_TIE_D)
            if lambd:
                args["train"]["fit"]["barlow"]["lambda"] = lambd
            configs[name] = args
    else:
        if ARGS.beta is None or ARGS.beta_proj is None:
            parser.error("full mode needs --beta and --beta_proj (choose them from the pilot).")
        out_dir   = os.path.join(here, "configs", "full")
        save_root = ARGS.save_root or "trained_nets/barlow"
        for c in CONDITIONS:
            beta = ARGS.beta_proj if c in ["c4", "c5"] else ARGS.beta
            for seed in ARGS.seeds:
                name = f"{c}_seed{seed}"
                configs[name] = make(name, c, beta, seed, save_root, ARGS.data_root)

    os.makedirs(out_dir, exist_ok=True)
    for name, args in configs.items():
        fn = os.path.join(out_dir, f"{name}.json")
        with open(fn, "w") as outfile:
            outfile.write(json.dumps(args, indent=4, sort_keys=True))
        bt = args["train"]["fit"].get("barlow")
        desc = "MSE only" if bt is None else \
            f"view_sigma={bt['view_sigma']}, beta={bt['beta']:g}, lambda={bt['lambda']:.3e}" + \
            (f", projector={bt['projector']['dims']}, n_pixels={bt['n_pixels']}" if "projector" in bt else "")
        desc += "".join(f", {k}={args['model'][k]}" for k in ["a_max", "tie_D"] if k in args["model"])
        print(f"{os.path.relpath(fn)}: {desc}")
    print(f"wrote {len(configs)} configs to {os.path.relpath(out_dir)}")

if __name__ == "__main__":
    main()
