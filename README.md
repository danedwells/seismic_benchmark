# Benchmarking seismic priors for earthquake early warning using the bEPIC location algorithm

> **Work in progress.** APIs, file layouts and workflows may change without notice. This repository is being developed with the assistance of LLM/AI tools.

A framework for evaluating the [bEPIC](https://github.com/danedwells/bEPIC) Bayesian earthquake early warning location algorithm across different spatial priors. For each real earthquake trigger sequence it runs bEPIC once per trigger version (each time a new station triggers), simulating real-time updates, and compares the posterior locations with USGS ANSS reference locations. bEPIC is currently the only supported algorithm.

Two regions are supported, sharing the same code: **California** (~700 events, 2008–2019) and **Cascadia**. Cascadia scripts are named `..._cascadia.py` and use `benchmark/config_cascadia.py`. Case studies (Ridgecrest, Ferndale, El Mayor, MTJ 2024 M7) are also included.

## Priors and workflows

| Prior | Source | Workflow |
|-------|--------|----------|
| `Gear1` | GEAR1 global hazard table | Time-independent |
| `NSHM` | USGS NSHM gridded + fault moment rates | Time-independent |
| `Helmstetter` | Helmstetter (2007) smoothed seismicity (needs pycsep) | Time-independent |
| `KDE_Seismicity` | KDE of historical seismicity from a USGS catalog (replaced `Smooth_seismicity`) | Time-independent |
| `Uniform` | Uninformative baseline | Time-independent |
| `ETAS` | Dynamic ETAS intensity λ(x, y, t), updated before each event | Time-dependent |
| `{prior}_etas_mixed` | Static prior blended with the ETAS prior (default α = 0.5) | Mixed |

- **Time-independent** (`time_independent_scripts/`): static priors, run in parallel.
- **Time-dependent** (`time_dependent_scripts/`): events run in chronological order; the ETAS prior is updated before each one and then fed the USGS location of the event just processed.
- **Mixed** (`mixed_prior_scripts/`): `combined = ALPHA * etas + (1 - ALPHA) * static`, on the ETAS grid; runs serially.

Results are written as CSVs, then plotted as maps and histograms.

## Dependencies and installation

Hard dependencies (install first, each with `pip install -e .`):

- [bEPIC](https://github.com/danedwells/bEPIC) — provides the location algorithm.
- [priors](https://github.com/danedwells/nehrp_priors) — `SeismicPrior`, `EtasPriorUpdater` and `.tt3` prior files.
- `pycsep` (only for the Helmstetter prior): `pip install git+https://github.com/SCECcode/pycsep.git`

Optional: [etas_2](https://github.com/danedwells/etas_2), only to re-run the ETAS inversion. A pre-inverted `data/california/etas_inversion/parameters_benchmark.json` is enough to run the benchmarks.

```bash
git clone git@github.com:danedwells/seismic_benchmark.git
cd seismic_benchmark
pip install -e .
pytest          # unit tests; no network or data files needed
```

## Directory structure

```
benchmark/                  # Installable package
  config_california.py, config_cascadia.py   # Prior filenames, bounds, benchmark/ETAS parameters (imported as `config`)
  runner.py                 # BenchmarkRunner and run helpers
  priors.py                 # Build/cache static priors; blend_priors()
  metrics.py, plots.py      # Error/calibration metrics; figure helpers
  usgs.py, background.py    # USGS/IRIS downloads, .run file builder; background seismicity
  parsing_tools.py          # EPIC log parsing
preparation_scripts/        # Run once first: static priors, ETAS inversion, .run files, station availability
time_independent_scripts/   # run_benchmarks.py, case_studies.py, run_cascadia.py
time_dependent_scripts/     # run_benchmarks.py, case_studies.py, run_cascadia.py, test_ideas/
mixed_prior_scripts/        # run_benchmarks.py, case_studies.py
log_scripts/                # Re-analysis of real EPIC log files
paper_figures/, plot_scripts/, data_examination_scripts/   # Figures, comparison plots, ad hoc inspection
Magna/                      # Self-contained Magna, UT sequence study
tests/                      # pytest unit tests
MARKDOWN/                   # Notes and planned experiments
data/                       # Inputs (not in git): {california,cascadia}/{run_files,etas_inversion,reference}, case_studies/, channel_availability/
results/                    # Outputs (not in git): {california,cascadia}/{output,figures}/{time_independent,time_dependent,mixed}, case_studies/
```

`data/` and `results/` are gitignored; you must supply the trigger-sequence and reference-catalog files separately.

## Prior source data

Not included because of size; contact daniel.wells@usu.edu if needed. Place files under `SeismicPrior.data_dir` at the paths in `PRIOR_CONSTRUCTION_PARAMS['source_paths']` (`benchmark/config_california.py`).

| Prior | Source |
|-------|--------|
| Gear1 | [GEAR1 paper](https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model) |
| NSHM | [gridded_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz), [fault_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/fault_moment_rates.xyz) (log₁₀ moment rates on the same 0.1° grid; exponentiated, summed, normalized) |
| Helmstetter | [PyCSEP artifact](https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat) / [paper](https://hal.science/hal-00195399/document) |
| KDE_Seismicity | Built from a downloaded USGS catalog; see `data_examination_scripts/examine_kde_prior.py` |
| ETAS | Generated by `preparation_scripts/build_initial_ETAS_prior_*.py` |

## Running

All scripts are Jupyter-style (cells delimited by `#%%`); run them cell by cell or top to bottom. Flags such as `RUN_ALL_PRIORS` and `SKIP_RUN` (load existing CSVs instead of running) sit at the top of each script.

**1. Preparation (once)** — scripts in `preparation_scripts/`, with a `_cascadia` variant where applicable:

```bash
python preparation_scripts/build_static_priors_california.py        # static .tt3 priors
python preparation_scripts/build_run_files_california_case_studies.py   # case-study catalogs + .run files
python preparation_scripts/build_initial_ETAS_prior_california.py   # ETAS inversion; slow, skip if parameters_benchmark.json exists
```

**2. Benchmarks**

```bash
python time_independent_scripts/run_benchmarks.py    # static priors
python time_dependent_scripts/run_benchmarks.py      # dynamic ETAS (tunables: ETAS_UPDATE_INTERVAL_S, PRIOR_ALPHA)
python mixed_prior_scripts/run_benchmarks.py         # blended priors (tunable: ALPHA)
```

Cascadia uses `run_cascadia.py` in the first two. The mixed workflow needs both the static priors and the ETAS inversion built first.

**3. Case studies** — `case_studies.py` in each workflow folder. Set `ACTIVE_CASE_STUDY` to `Ridgecrest`, `Ferndale`, `ElMayor` or `MTJ_2024_M7`; the case-study preparation script must have run first. Results go to `results/case_studies/{name}/`.

Outputs for the main benchmarks go to `results/<region>/output/<workflow>/max_trigs_{N}/` and `results/<region>/figures/<workflow>/max_trigs_{N}/`.

## EPIC log re-analysis

`log_scripts/` compares logged real-time EPIC solutions with bEPIC replays (station-count rules, station masking, ETAS replay). It builds on `benchmark/parsing_tools.py`; see `MARKDOWN/AMY_SCRIPTS_REVIEW.md` for details. Channel availability files are in `data/channel_availability/`.

## Output format

Each `{prior}_benchmark_results.csv` has one row per (event, trigger version):

| Column | Description |
|--------|-------------|
| `event_id`, `version` | Event ID (`.run` filename stem) and trigger version |
| `posterior_lat`, `posterior_lon` | bEPIC posterior location |
| `best_misfit`, `best_like`, `best_prior`, `frac_misfit` | Misfit, likelihood and prior at the best location |
| `map_err_km` | Distance from the posterior location to the USGS location |
| `posterior_confidence_level` | Smallest credible region (0–1) containing the USGS location; uniform across events if the posterior is calibrated |

Other diagnostic columns (likelihood-only and prior-only variants, posterior coverage at fixed radii) are mainly used by `plot_scripts/`.

## Design notes

- `priors`, `etas_2` and `bEPIC` remain separate repositories and may be absorbed later.
- The three workflows are separate folders because the time-dependent one is inherently sequential (each event's location changes the next prior), even though the scripts overlap.
- Metrics: location error (km), posterior coverage within 10/25/50/100 km, and `posterior_confidence_level`.

## Known limitations

- Waveform processing and per-station magnitude calculation are not yet implemented.
- The high-resolution `REFERENCE` workflow is disabled.
- The `BenchmarkRunner` API may change; the test suite is minimal.
