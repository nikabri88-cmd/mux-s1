# MUX-S1: what do the data of Hesse et al. (2026) draw?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23738757.svg)](https://doi.org/10.5281/zenodo.23738757)

A pre-registered study of the population dynamics behind the rapid alternation of "conscious" and
"suppressed" micro-epochs reported by Hesse, Lanfranchi, McGrory, Shi, Engel and Tsao
(bioRxiv [10.64898/2026.10.15.759299](https://doi.org/10.64898/2026.10.15.759299), posted 16 October 2026)
in macaque face patches during binocular flash suppression.

Authors: Julia Ryzhkova (Siriona, siriona.one; ORCID 0009-0005-4194-6059) and Claude (Anthropic).
Status: version 1.1 complete, 28 October 2026. Archived on Zenodo: https://doi.org/10.5281/zenodo.23738757 (release v1.1.0).

## Two versions

| Version | What | Fixed | Archive (SHA-256) |
| --- | --- | --- | --- |
| 1.0 | Simulations before the data release: which latent dynamics fit the published fingerprint, and which unpublished statistics would tell them apart | 18 October 2026 | `archives/MUX-S1_v1.0.zip` (`a82b428d…`) |
| 1.1 | The check of the 1.0 predictions against the authors' released data: thresholds, code and reading rules fixed before the release | 21 October 2026, hashes published in a dated post | `archives/mux_s1_v11.zip` (`5df58997…`) |
| 1.1, pre-run record | Reader of the release, face features, B classes, declared secondary analyses; written before A and B were computed | 28 October 2026 | `archives/mux_s1_v11_prerun.zip` (`b061553d…`) |
| 1.1, results | Primary and secondary runs, exploratory run, deviation log, report | 28 October 2026 | `archives/mux_s1_v11_final.zip` (`4d2728ce…`) |

Full hashes are in `MUX-S1_prereg.sha256`, `MUX-S1_v11.sha256` and the running log
`MUX-S1_v11_adapter.sha256`. The dates of the pre-registration rest on the dated posts and the
timestamps in these logs; a SHA-256 by itself proves content, and the date comes from where it was
published.

## Version 1.1: results on the released data

The authors' data were released on Zenodo on 26 October 2026 (record 23604909). The primary run is
the frozen `verify_real.py` on the 9 sessions of the second experiment (FACE1500): 6740 neurons,
7000 stimulus pairs. Only this run gives verdicts.

### Calibration

The calibration targets A1–A4 (statistics read from the authors' figures) failed on the full set of
pairs: A1 0.107, A2 0.424, A3 0.376, A4 deviation 0.52. Following reading rule 5, the gap was checked
first as a difference of methods. The secondary run declared in advance on the repeated stimuli only
(30 pairs per session, about 20 trials per condition) reproduced the authors' figures: A1 0.245,
A3 0.492, A4 deviation 0.10 (dwell curve 60 / 80 / 100 ms against 62 / 77 / 111 ms), D 64.5 ms against
the published 67 ms. A2 was 0.485, above its range.

The most likely explanation: templates built from pairs with one or a few trials are noisy, and the
state labels fragment. This explanation is inferred: the true state labels are unknown, and the
calibration shows only the mismatch with the expected statistics.

### Verdicts of the primary run

| Prediction | Verdict | Value | Class |
| --- | --- | --- | --- |
| B1 HMM median dwell / D | confirmed | 0.465 | reproduction |
| B2 D with a global mean / D with a per-trial mean | confirmed | 6.19 (lower bound: the fit reached its 400 ms ceiling) | independent |
| B3 ACF amplitude / 0.224 | confirmed | 1.89 | follows from A2 |
| B4 two-rate gain against the number of label neurons | refuted | −0.0021 at 10, −0.0001 at half | independent |
| B5 Fig. 3a at equal estimation noise | refuted | share 0.523 at threshold 0.50 | independent |
| B6 decoder without the mean rate | confirmed by construction (see the deviation log) | 0.500 (raw 0.593) | consequence of the construction |
| B7 gain, early minus late window | confirmed | +0.135 (1.459 and 1.324) | independent |
| B9 share of cells with a higher rate in state 2 | refuted | 0.476 at ratio 1.389 | mixed |
| B10 gain against R² | refuted | rho 0.105, drop −0.075 | independent |
| B11 CV of dwell times | refuted | 1.214, "smooth mixing" band | independent |
| B12 trial overdispersion | refuted | 1.864 | independent |

Confirmed 5, refuted 6, not computed 0. The classes were fixed before the run
(`MUX-S1_v11_B_classes.md`); B6 was reclassified after the run, as logged.

### Exploratory run E-repeat (after the primary result; gives no verdicts)

The frozen counts block on the 30 repeated pairs per session, where the calibration matched:

| | primary | E-repeat |
| --- | --- | --- |
| B1 | 0.465 | 0.930 (the value read from the figures) |
| B2 | 6.19 | 3.62 |
| B3 | 1.89 | 2.17 |
| B4 | −0.0021 / −0.0001 | −0.0005 at 10, +0.0006 at half, monotone: holds |
| B5 | share 0.523 | share 0.285: holds |
| B11 | 1.214 | 0.996: the refutation stays |
| B12 | 1.864 | 1.138: holds |

On the subset where the calibration reproduces, the refutations of B4, B5 and B12 disappear and the
refutation of B11 stays. B9 and B10 need hundreds of faces, so this subset cannot test them.

### What it means

1. **The time scale is there.** The LLR autocorrelation gives D 64.5 ms on both samples against the
   published 67 ms: alternation of the order of 15 times a second reproduces in the data.
2. **The rate depends on the estimator.** The ratio of D with a global mean to D with a per-trial
   mean is 3.6 on the repeated pairs (6.2 on the full set, where the fit hits its ceiling) against 1.7
   for the simulated Markov model. This establishes sensitivity to centering; the source of the
   trial-to-trial variance (stimuli, conditions, drift, On/Off fluctuations) is left open by this run.
3. **B6 is withdrawn as evidence.** Our control with per-state means subtracted is degenerate for a
   linear order decoder and gives 0.5 for any difference between states (a design flaw of ours, found
   by an external review after the run). It limits only this analysis and says nothing about the
   authors' decoder.
4. **The authors' key test (Fig. 3a) is sensitive to how the static alternative is specified.** With
   a static model built from the same priming templates, 29% of the gain remains on the calibrated
   subset and 52% on the full set, and the gain stays positive. This control changes both the noise
   and the model class, so the remainder is attributed to neither alone.
5. **The strict switching test gives a weak signal of the right shape** (B4 on E-repeat): the
   advantage of two rates appears only with enough label neurons.
6. **Dwell times are more irregular than Markov** (B11, CV about 1.0 against the band 0.75–0.90). The
   band was set on our processed Markov simulation, so it is specific to that simulation; label noise
   raises the CV and is smaller on 30 pairs, though still present.
7. **The gain asymmetry is stronger in the early window** (B7): the stimulus history shows in it.

In sum: the alternation is in the data, and its mechanism is open. The evidence for discrete
switching is weaker than the paper reads: the gain of the main test depends on the static
alternative, and the spread of dwell times lies outside the band of the processed Markov simulation.
Of the five confirmed predictions, B2 and B7 carry content, both with the caveat on label calibration
on the full set.

### Injections

Comments about injections of saline and then psilocybin appear in four sessions: 250504_Mowgli,
250518_Mowgli, 251212_Alfie, 251218_Alfie. The last two belong to FACE1500; in their rivalry paradigm
no trial follows an injection comment (the comments carry an empty next-trial index), and 251212 has
the note "Darkness sober". In 250504_Mowgli the last 2 trials follow the injection. The injections are
mentioned neither in the authors' code nor in the preprint text.

Details: `MUX-S1_v11_results.md` (report, Russian) and `MUX-S1_v11_deviations.md` (full deviation log,
Russian).

## Version 1.0: simulations before the release

Synthetic face-cell populations (linear axis tuning in a 50-feature face space, Poisson spikes in
10-ms bins) are driven by a latent weight of the conscious face. Candidate shapes and mechanisms:

| Model | Latent dynamics / mechanism |
|---|---|
| M0 | two-state Markov switching (the authors' model) |
| M1, M1b | rhythmic switching, gamma-distributed dwell times (k = 6, k = 3) |
| M2 | continuous unimodal mixing (logit Ornstein-Uhlenbeck, SD 1.0) |
| M3 | continuous bimodal mixing (logit Ornstein-Uhlenbeck, SD 2.5) |
| M4, M4c | M0 plus stimulus adaptation of the whole drive / of the tuning part only |
| M5 | M0 plus a 1.21x tuning gain of the conscious face |
| M6 | M0 with 20% of trials where the flash fails |
| M7 | a ring: C -> X -> S -> Y -> C through two different intermediate patterns |
| C1, C2 | the authors' controls: averaging and uncoordinated switching |

The authors' analysis pipeline is reimplemented (templates from priming, Poisson LLR, fixed-emission
HMM with Baum-Welch transitions and Viterbi paths, PCA + OLS face reconstruction with leave-one-face-out
weights, modulation indices, tuning gain, order-concatenation state decoder, held-out likelihood test)
together with new discriminating statistics.

Main results on synthetic data (details in `MUX-S1_report.md`, Russian):

- The authors' controls (averaging, uncoordinated switching) fail 3 of the 4 tests of their battery, as in the paper; circular reconstruction is passed by every model.
- The held-out likelihood test (Extended Data Fig. 10f) rejects continuous mixing, unimodal and bimodal, at every signal strength tested. With hard state labels from half of the neurons it acts as a threshold test: two rates beat the best static mixture only when the labels are right in more than about 75% of bins.
- The sign of the Fig. 3a test is won by every model, including models without switching, because the two sides carry unequal estimation noise.
- The shape of the published LLR autocorrelation matches Markov switching and rules out strong rhythmicity among the tested models.
- A tuning-gain ratio of about 1.2 is produced by plain stimulus adaptation; no tested model reproduces it together with the higher mean rate in the suppressed state that the authors report.
- Pre-registered verdicts: H2, H3, H5 and H7 confirmed; H1, H4 and H6 not confirmed. Three discriminating statistics (D1, D2, D5) had design flaws; corrected versions are reported as exploratory analyses (E4, E5).

## Files

Version 1.0
- `MUX-S1_prereg.md` – pre-registration (Russian); `MUX-S1_prereg.sha256`, `MUX-S1_prereg_manifest.txt`, `MUX-S1_analysis.sha256`, `MUX-S1_calibration.sha256` – hashes
- `MUX-S1_deviations.md`, `MUX-S1_report.md` – deviation log and report (Russian)
- `mux_s1/core.py`, `experiments.py`, `analysis.py`, `runner.py`, `tests.py` – simulator, analyses, tests
- `calibrate.py`, `run_parts.py`, `analyze.py`, `explore_*.py`, `analyze_explore.py`, `figures.py`, `fill_*.py`, `report_numbers.py` – runs and report
- `digitize/` – digitization of Figs. 2c, 2d and 5c of the preprint (`targets.json`)
- `results/` – calibration, `partI/`, `partII/`, `summary.json`; `figures/` – figures
- `review_brief.md` – the brief given to the independent reviewer of 1.0

Version 1.1
- `MUX-S1_v11_prereg.md` – pre-registration (Russian); `MUX-S1_v11.sha256` – its hashes
- `mux_s1/realdata.py`, `verify_real.py`, `mux_s1/tests_v11.py`, `hashes_v11.py` – the frozen checks
- `mux_s1/hesse_reader.py` – reader of the released MATLAB v7.3 files (trial types, unit list, dominant face)
- `mux_s1/faces.py`, `mux_s1/feature_maps.npz` – face features for the gain block
- `mux_s1/adapters.py` – from the reader to the canonical form of the checks
- `mux_s1/tests_hesse.py`, `tests_faces.py`, `tests_adapter.py` – tests of the reader, features and adapter
- `verify_secondary.py` – the secondary analyses declared before the run; `explore_repeat.py` – the exploratory run E-repeat
- `MUX_S1_step1_structure.ipynb` … `MUX_S1_step4_explore.ipynb` (sources `colab_step*.py`) – the Colab notebooks that ran each step; `mux_s1_step*_code.zip` – the exact code bundle each notebook checked by hash before running
- `MUX-S1_v11_B_classes.md` – what each verdict means, fixed before the run
- `MUX-S1_v11_post_prerun.md` – the post published before the run
- `MUX-S1_v11_deviations.md`, `MUX-S1_v11_results.md` – deviation log and report (Russian)
- `MUX-S1_v11_adapter.sha256` – running hash log of everything written after the pre-registration
- `results/real/` – primary (`primary/`), secondary (`secondary/`) and exploratory (`explore/`) outputs, with logs; `mux_s1_v11_results_32e5655e9f6c.zip` is the archive downloaded from the run
- `archives/` – the four published archives listed above

## Reproduce

```bash
pip install -r requirements.txt

# tests
python -m mux_s1.tests            # 1.0
python -m mux_s1.tests_v11        # 1.1, frozen checks
python -m mux_s1.tests_hesse
python -m mux_s1.tests_faces
python -m mux_s1.tests_adapter

# 1.0, synthetic
python calibrate.py
python run_parts.py I
python run_parts.py II
python analyze.py
python figures.py

# 1.1, the frozen checks on synthetic data (the pre-release test)
python verify_real.py --source synthetic --model M0_markov
```

The runs on the released data were done in Google Colab with the notebooks, in order: step 1
(structure of the files), step 2 (downloads the session files from Zenodo one at a time, reads all
19 sessions and caches spike counts), step 2b (face features), step 3 (primary and secondary runs; needs a High-RAM
runtime, about 51 GB), step 4 (E-repeat). Outside Colab, the same commands are:

```bash
python verify_real.py --source dandi --path <cache dir> --design FACE1500 --out results/real
python verify_secondary.py --cache <cache dir> --out results/real/secondary
python explore_repeat.py --cache <cache dir> --out results/real/explore
```

Tests pass with Python 3.13, numpy 2.5, scipy 1.18, scikit-learn 1.9, matplotlib 3.11 and h5py 3.16;
the release runs used Colab with its default packages. `mat-io` (numpy 2.2.2 or later) is needed only
to rebuild `feature_maps.npz` from the authors' files.

## Sources and licences

- **Code** (`mux_s1/*.py`, `*.py`, notebooks): MIT, see `LICENSE`.
- **Text, figures, results and digitized data** (`*.md`, `figures/`, `results/`, `digitize/*.json`): CC BY 4.0.
- **The authors' data** (Zenodo record [23604909](https://zenodo.org/records/23604909), CC BY 4.0) are not redistributed, with one exception: `mux_s1/feature_maps.npz` holds the face feature maps of the two stimulus sets (`data.zip`, `data/stimuli/`, files `feat_map_priming_reconstructed_0.00.mat` for liang_2000 and liang_2500), converted to NumPy format with values unchanged; the SHA-256 of each source file is stored inside. Credit: Hesse, Lanfranchi, McGrory, Shi, Engel, Tsao (2026), CC BY 4.0.
- **The authors' code** ([miou/conscious_suppressed_alternations](https://github.com/miou/conscious_suppressed_alternations)) carried no licence file when checked on 28 October 2026, and none of it is copied here. `mux_s1/hesse_reader.py` is an independent Python implementation of the trial, unit and stimulus rules described in their MATLAB code; each rule names the file it comes from.
- **Figure crops** in `digitize/` come from the preprint (CC BY 4.0). The preprint itself is not included: download it from bioRxiv (DOI above) to `digitize/hesse2026.pdf` to rerun the digitization.

## Citation

See `CITATION.cff`. Related work: Ryzhkova, J., Theory of Variability, v2.7,
[10.5281/zenodo.19627721](https://doi.org/10.5281/zenodo.19627721).
