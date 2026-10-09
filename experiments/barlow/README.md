# GDLNet + Barlow Twins: dictionary diversity experiments

Question: does adding a Barlow Twins loss on GDLNet's final sparse code make the
learned (analysis) dictionaries less redundant, without hurting denoising?

## Setup

Model: GDLNet-S paper settings, Gabor mixture order 1, untied layers:
`K=30, M=169, P=11, s=2, adaptive=True` (70,980 params).

Training: CBSD432 (128x128 crops, batch 10), validation on Kodak, test on CBSD68 (grayscale),
`sigma ~ U[10,40]` per sample (all conditions), Adam lr 1e-3, StepLR(50, 0.95),
clip_grad 5e-2, 6000 epochs, backtracking at 2 dB.

Loss for two-view conditions (`train.fit.barlow`, see `train.py`):

    L = 1/2 (MSE(x_hat_1, x) + MSE(x_hat_2, x)) + beta * L_BT(z_1, z_2)

where `z_i` is the final sparse code of noisy view `y_i = x + n_i`. `L_BT` is computed over
code pixels (rows) and subbands (features), normalized by the feature dim D.
The off-diagonal weight is scaled with the feature dim, `lambda = 5e-3 * M / D`
(5e-3 without projector, 4.126e-4 with the 2048-d projector).

| Cond. | Views | View sigma | beta | Projector |
|---|---|---|---|---|
| c0 | 1 | - | - | - |
| c1 | 2 | same | 0 | - |
| c2 | 2 | same | beta* | - |
| c3 | 2 | diff (resampled) | beta* | - |
| c4 | 2 | same | beta*_proj | 3x2048, 16384 sampled pixels |
| c5 | 2 | diff (resampled) | beta*_proj | 3x2048, 16384 sampled pixels |

c1 isolates the effect of seeing two noise draws per image from the effect of the Barlow term.
Validation/test always use a single noisy view, so PSNR is comparable across conditions.

## Metrics

Primary (analysis filters `A_k`, averaged over k), from `model/metrics.py`:
- `mu_s`: max off-diagonal shift-coherence `max_tau |<a_i, S_tau a_j>|` of unit-norm atoms.
- `n_dup_pairs`: number of atom pairs with shift-coherence > 0.9.

Also reported: mean shift-coherence, zero-shift coherence `mu_0` (separates true duplicates
from shifted/quadrature pairs), the same metrics for the final dictionary `D`, and test PSNR
at sigma in {5, 15, 25, 35, 50} (5 and 50 are outside the training range).

Note: the Barlow term sends no gradient to `D` (it only appears in `x_hat = D z`), so the
analysis filters are where its effect is expected.

## Workflow

All commands from the repo root. `analyze.py` needs PyWavelets installed.

**1. Smoke test (local, CPU, seconds):**

    python experiments/barlow/make_configs.py --smoke
    for f in experiments/barlow/configs/smoke/*.json; do python train.py $f; done

Runs go to `experiments/barlow/smoke_runs/` (git-ignored).

**2. Pilot: choose beta\*.** c2 and c4 for beta in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, plus c0 and c1
references, 300 epochs, seed 0 (12 runs, `trained_nets/barlow/pilot/`):

    python experiments/barlow/make_configs.py --pilot
    python train.py experiments/barlow/configs/pilot/<run>.json

Pick beta\* (and beta\*_proj) from the PSNR vs `mu_s` / `n_dup_pairs` trade-off: the largest
beta whose val/test PSNR stays within ~0.1 dB of c1.

**3. Full runs:** c0-c5 x seeds {0, 1} (12 runs, `trained_nets/barlow/`):

    python experiments/barlow/make_configs.py --beta <beta*> --beta_proj <beta*_proj>
    python train.py experiments/barlow/configs/full/<run>.json

**4. Analysis per run** (uses the `args.json` + `net.ckpt` that training writes to the run dir):

    python analyze.py trained_nets/barlow/<run>/args.json --test dataset/CBSD68/ --noise_level 5 15 25 35 50 --coherence

Keep the trailing slash on `dataset/CBSD68/`: `analyze.py` names the log after the parent
dir of the path, so this gives `test_CBSD68_None.txt` (without it: `test_dataset_None.txt`).

**5. Aggregate** (every subdirectory with an `args.json` is a run; groups are (condition, beta),
seeds pooled as mean ± std):

    python experiments/barlow/aggregate.py trained_nets/barlow/pilot
    python experiments/barlow/aggregate.py trained_nets/barlow

Writes to `<runs_root>/aggregate/`:

| File | Content |
|---|---|
| `summary.md` | table per group: μ_s, n_dup, μ_0 of A (mean over k), μ_s, n_dup of D, val/test PSNR, mean C_ii, backtracks |
| `runs.csv`, `groups.csv` | the same numbers per run / per group |
| `tradeoff.png` | test PSNR (σ=25, `--psnr_sigma`) vs μ_s(A) and vs n_dup(A); beta sweeps connected (pilot) |
| `training.png` | n_dup(A), μ_s(A) over training, one column per Barlow condition, c0/c1 as gray references |
| `layers.png` | final μ_s(A_k), n_dup(A_k) per layer k, same layout |

Runs without `coherence/coherence.json` fall back to their last `dict_metrics.jsonl` entry, and
runs without a test log fall back to final val PSNR (both are reported when it happens).

## Per-run outputs

| File | Content |
|---|---|
| `train.txt`, `val.txt`, `test.txt` | PSNR per epoch (train PSNR is from the MSE term only) |
| `barlow.txt` | per-epoch train averages: mse, barlow, on, off, mean C_ii (two-view conditions) |
| `dict_metrics.jsonl` | dictionary metrics at epoch 0 and every `val_freq` epochs (an epoch can repeat after a backtrack or resume; use the last entry) |
| `backtrack.txt` | epochs at which training backtracked |
| `coherence/` | from `analyze.py --coherence`: `coherence.json`, heatmaps, histogram, per-layer curves |
| `test_CBSD68_None.txt` | from `analyze.py --test`: `sigma, PSNR` per line |

## Caveats

- **Resuming** (`python train.py <run>/args.json`, whose `ckpt` points at `net.ckpt`) trains for
  another `epochs` epochs starting after the checkpoint epoch, not up to `epochs`. Set
  `train.fit.epochs` in that `args.json` to the number of remaining epochs before resuming.
- Two-view conditions cost ~2x compute and activation memory per step relative to c0.
- cuDNN is not forced deterministic, so same-seed runs share their initialization but
  are not bit-identical on GPU.
