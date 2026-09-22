# Limits of Manifold Projection in Calibrated Channel and Clutter Estimation

Simulation and measurement code for the paper

> A. Entezami and L. Dong, "Limits of Manifold Projection in Calibrated Channel
> and Clutter Estimation," submitted to *IEEE Transactions on Signal Processing*.

Every number, table entry and plotted point in the paper is produced by a script
in `sim/`. The map below says which. Nothing here is hand-entered.

The manuscript sources are **not** in this repository.

---

## Quick start

```bash
python -m pip install -r requirements.txt
python data/fetch_data.py --help          # see data/README.md first
OMP_NUM_THREADS=4 python sim/table_inspan.py
```

Run everything from the repository root; the scripts expect `sim/` on the path
and `data/` and `figs/data/` relative to the working directory.

**Set `OMP_NUM_THREADS=4`.** Unconstrained BLAS oversubscribes on many machines
and turns a three-second job into tens of CPU-minutes.

---

## Where each result comes from

### Figures

| Paper | Generator | Exported data |
|---|---|---|
| Fig. 1 | none — TikZ schematic, no data | — |
| Fig. 2, operating region | `sim/fig_C_operating_region.py` | `figs/data/figC_locus.dat`, `figC_marks.dat` |
| Fig. 3, projection gain | `sim/fig_A_projection.py` | `figs/data/figA_pred_*.dat`, `figA_mc_*.dat`, `figA_diffuse.dat` |
| Fig. 4, sensing accuracy | `sim/fig_B_sensing.py` | `figs/data/figB_{diffuse,noise}_{delay,angle}.dat` |
| Fig. 5, measured channels | `sim/exp_real_rproj.py` | `figs/data/figD_real.dat`, `figD_marks.dat` |

`figB_*.dat` columns are `x raw rlo rhi str slo shi shr hlo hhi crb`: the three
estimators with their 95% bootstrap intervals, then the calibration-aware bound
of Proposition 5 computed on the same target draws.

### Tables

| Paper | Generator |
|---|---|
| Table I, reference configuration | background energies printed by `sim/verify_akram_bounds.py` |
| Table II, in-span fraction | `sim/table_inspan.py` |
| Table III, measured residual shares | `sim/table_real.py` |

### Quantities quoted in the text

| Section | Claim | Script |
|---|---|---|
| II | a 1° atom error leaves 0.3% of that atom out of span | `sim/verify_round2_claims.py` |
| IV | combined gain `(1+ρ)/(ρ+κ+ν)` matches Monte Carlo to 2.5% | `sim/verify_akram_bounds.py` |
| IV | per-chain drift deposits only ~47% of its energy in span(B) | `sim/verify_round2_claims.py` |
| IV | the full-support example needs *independent* angle and delay | `sim/verify_round2_claims.py` |
| IV | Table II quadrature law vs simulator law, gap 0.00093 | `sim/verify_round2_claims.py` |
| V | 3 dB bound on the deterministic persistence weight | `sim/verify_akram_bounds.py` |
| V | plug-in vs that bound: +0.1% isotropic, +6.6% rank-one | `sim/verify_two_epoch_plugin.py` |
| V | scalar weight gives up 4.3 dB to the matrix Wiener filter | `sim/verify_general_cov.py` |
| V | uniform dominance of the positive-part plug-in | `sim/verify_dominance.py` |
| VI | payload-only, calibration-aware and clairvoyant bounds | `sim/crb_calaware.py` |
| VI | payload-only Schur complement identity | `sim/crb_nuisance.py` |
| VI | collinear amplitude floor | `sim/exp_collinear.py` |
| VI | the floor binds unbiased estimators only | `sim/verify_round2_claims.py` |
| VII-B | measured residual shares and order sensitivity | `sim/table_real.py` |
| VII-B | receiver noise variance from the delay-domain floor | `sim/noise_floor.py` |
| VII-B | which offset convention is applied, and its cost | `sim/verify_offset_convention.py` |
| VII-B | what the 32 selected snapshots cover | `sim/verify_offset_convention.py` |
| VII-B | the 0.074 λ decorrelation is an artefact (control) | `sim/verify_coherence.py` |
| VII-B | delay-compensated coherence 0.95 at one step | `sim/verify_sto_drift.py` |

### Libraries

`sim/isac_core.py` holds the signal model, the dictionary and the three
background estimators. `sim/tfrecord_reader.py` is a dependency-free TFRecord and
protobuf reader. `sim/exp_real_dichasus.py` and `sim/exp_real_env2.py` load the two
measured datasets and run the orthogonal matching pursuit; both are also runnable
on their own.

---

## Two conventions worth knowing before reading the code

**Diffuse fields are scaled to a prescribed *ensemble* energy, not to a fixed
energy per realization.** Because every atom has `‖b(φ,ξ)‖² = N`, the correct
scale is deterministic. Renormalizing each draw couples the coefficients to the
geometry drawn with them, breaks the independence hypothesis of Proposition 2, and
lowers the measured in-span fraction from 0.710 to 0.687 no matter how many trials
are run. See the comment in `isac_core.draw_diffuse`.

**Every Cramér–Rao inverse is taken after nondimensionalizing** by amplitude,
microseconds and radians. The Fisher matrix mixes those units, so its singular
values span many decades and `numpy.linalg.pinv` silently truncates the weak
direction of a near-collinear draw. `sim/crb_calaware.py` carries a
`unit_invariance()` guard that asserts the converted bound does not depend on the
unit chosen for the delay.

---

## Script status

Most scripts support the current manuscript. Two do not, and are kept for the
record rather than for reuse:

- **`sim/verify_operating_point.py` — retracted analysis, do not cite.** It maps
  published reciprocity-calibration stability figures onto the drift fraction ρ.
  That mapping is wrong: a residual per-chain coefficient acts as `diag(e)Bg`, not
  as `Bδ`, and deposits only about 47% of its energy inside the span. The paper
  sweeps ρ instead of measuring it, and explains why.
- `sim/exp_sensing.py` and `sim/verify_drift.py` are earlier versions of the
  sensing experiment and of the drift check, superseded by `fig_B_sensing.py` and
  `verify_akram_bounds.py` respectively.

`sim/exp_two_epoch.py` measures the cross-epoch residual correlation on real
channels. Read its output together with `verify_coherence.py` and
`verify_sto_drift.py`: the uncorrected statistic is dominated by the sounder's
burst-alignment granularity, and the paper does not claim to identify `C₀₁`.

---

## Environment

Developed with Python 3.12.13, NumPy 2.2.6, SciPy 1.17.1 and Matplotlib 3.10.8 on
Linux. `requirements.txt` pins those. The figure generators write both the `.dat`
files consumed by the paper's pgfplots sources and a standalone PDF preview.

## Data

The measured channels are third-party and **nothing from them is redistributed
here**, neither the recordings nor the small metadata files: we do not hold a copy
of their licence terms. `data/README.md` gives the two datasets, their DOIs, the
files you need to obtain, their checksums, and the exact byte range, record indices
and sub-array the paper uses. `python data/fetch_data.py --verify` confirms you have
the same files we did.

What *is* published here are our own results computed from those recordings —
`figs/data/figD_*.dat` and the Table III shares — which is the same content the
paper itself reports, not a republication of the data.

## Licence

Code in this repository: MIT, see `LICENSE`. The measured data is covered by its
own licence, recorded in `data/README.md`.
