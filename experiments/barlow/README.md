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
- `n_dup_pairs`: number of atom pairs with shift-coherence `max_tau |<a_i, S_tau a_j>|` > 0.9.
- `n_dup_pairs_nc`: the same, among non-collapsed atoms only.
- `n_collapsed`: atoms whose energy std is < 0.8 px along every axis (`effective_support`):
  spikes and 1-3 px blobs, which are near-identical whatever their Gabor frequency/phase.
- `n_dead`: subbands active on < 1e-3 of code coefficients (`code_activity`), on the
  `analyze.py --test` images at sigma=25 (`--activity_sigma`), or on the validation images at the
  mid-range sigma during training.

Also reported: `mu_s` (max shift-coherence; saturated at ~0.998 in the pilot, so not a headline),
mean shift-coherence, zero-shift coherence `mu_0` (separates true duplicates from shifted/quadrature
pairs), the same metrics for the final dictionary `D`, and test PSNR at sigma in {5, 15, 25, 35, 50}
(5 and 50 are outside the training range).

Envelope clamp: `model.a_max` bounds the Gabor envelope precision `|a| <= a_max` per axis
(at init and in `project()`), i.e. envelope energy std >= ~1/(2 a_max) px. Clamped `randn` init puts
many atoms at the bound, so pick `a_max` with margin below `1/(2 * 0.8) = 0.625` (e.g. 0.5 -> >= 1 px)
for clamped runs to have no collapsed atoms by the metric.

Tying to D: `model.tie_D` (e.g. `["a", "w0", "psi"]`) ties those Gabor shape parameters of every
`A_k` (incl. `A_0`) and `B_k` to the dictionary `D`, leaving the scale `alpha` free per operator:
`A_k = alpha_A,k * D_shape`, `B_k = alpha_B,k * D_shape` (ISTA-like with learned per-layer gains;
21,125 params instead of 70,980). Shift-coherence metrics of every `A_k` then equal those of `D`, and
the Barlow loss, which sends no gradient to an untied `D`, reaches `D`'s shapes through the `A_k`.
Not combinable with the older `shared` option, which ties `A_k` to `A_0` and `B_k` to `D` separately.

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

**2b. Pilot 2: envelope clamp and tying to D.** Pilot 1 found the redundancy dominated by collapsed
(spike/blob) atoms, and no Barlow effect on `D`. 300 epochs, seed 0, beta=1e-3 for c2
(12 runs, `trained_nets/barlow/pilot2/`):
- c1, c2 x `a_max` {none, 0.5} x `tie_D` {none, [a, w0, psi]} (8 runs; `pilot2_c1`, `pilot2_c2`
  re-run pilot 1's `pilot_c1`, `pilot_c2_beta0.001` with identical configs: run-to-run noise floor),
- clamped c2 at lambda {0.05, 0.5}, untied and tied (4 runs; with the lambda=5e-3 runs, a 3-point sweep).
  Larger lambda also scales the whole Barlow term (about 12x at lambda=0.5 at init), not only its
  off-diagonal share.

      python experiments/barlow/make_configs.py --pilot2
      python train.py experiments/barlow/configs/pilot2/<run>.json

**3. Final runs:** 5 arms chosen from pilot 2 x seeds {0, 1}, 6000 epochs
(10 runs, `trained_nets/barlow/final/`). beta=1e-3 for c2, `a_max`=0.5, `tie_D`=[a, w0, psi]:

| Arm | Barlow | `a_max` | `tie_D` | lambda | Params |
|---|---|---|---|---|---|
| `c0` | - (single view, MSE; paper setting) | - | - | - | 70,980 |
| `c1_amax0.5` | beta=0 (2 views) | 0.5 | - | - | 70,980 |
| `c2_amax0.5_lam0.05` | beta=1e-3 | 0.5 | - | 0.05 | 70,980 |
| `c1_amax0.5_tieD` | beta=0 (2 views) | 0.5 | a, w0, psi | - | 21,125 |
| `c2_amax0.5_tieD` | beta=1e-3 | 0.5 | a, w0, psi | 5e-3 | 21,125 |

Each config equals its pilot counterpart except seed, epochs and save path. Within a seed, all
arms start from the same initial dictionary.

    python experiments/barlow/make_configs.py --final
    python train.py experiments/barlow/configs/final/<run>.json

(The original c0-c5 full mode, `make_configs.py --beta B --beta_proj BP`, is superseded by `--final`.)

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
