# Benchmarking seismic priors for earthquake early warning using the bEPIC location algorithm

> **Work in progress.** This repository is under active development. APIs, file layouts, and workflows may change without notice. This repository is being developed with the assistance of LLM/AI tools.

A benchmarking framework for evaluating the [bEPIC](https://github.com/danedwells/bEPIC) Bayesian earthquake early warning location algorithm across different spatial prior distributions. Given a set of real earthquake trigger sequences, it runs bEPIC iteratively as station triggers arrive and compares the resulting posterior locations against USGS ANSS catalog reference positions.

Two regions are supported: **California** (the main, ~700-event benchmark catalog) and **Cascadia** (Pacific Northwest). Both share the same code and workflows below; Cascadia-specific scripts are named `..._cascadia.py` and its config lives in `benchmark/config_cascadia.py`.

Three parallel workflows are provided:

- **Time-independent** (`time_independent_scripts/`) — five static spatial priors, unchanged per-event, evaluated in parallel.
- **Time-dependent** (`time_dependent_scripts/`) — a dynamic ETAS prior that evolves as events are located, capturing the time-varying spatial distribution of aftershock hazard.
- **Mixed** (`mixed_prior_scripts/`) — each of the five static priors linearly blended with the live ETAS prior, combining long-term background rates with short-term aftershock clustering.

---

## What it does

For each event in the test catalog, bEPIC is run once per trigger version (i.e., once each time a new station triggers), simulating real-time location updates. This is repeated across spatial priors:

| Prior | Source | Workflow |
|-------|--------|----------|
| `Gear1` | GEAR1 global seismic hazard table | Time-independent |
| `NSHM` | USGS NSHM gridded + fault moment rates (summed in linear space) | Time-independent |
| `Helmstetter` | Helmstetter (2007) smoothed seismicity (requires pycsep) | Time-independent |
| `KDE_Seismicity` | Smoothed historical seismicity, built from a downloaded USGS catalog (replaces the older `Smooth_seismicity` prior) | Time-independent |
| `Uniform` | Uninformative baseline; equivalent to having no prior | Time-independent |
| `ETAS` | Dynamic ETAS conditional intensity λ(x, y, t), updated before each event | Time-dependent |
| `{prior}_etas_mixed` | Linear blend of each static prior with the dynamic ETAS prior (default α = 0.5) | Mixed |

Results (posterior lat/lon, travel-time misfits, location errors vs. ANSS) are written to CSV and visualized as maps and histograms. Each run also produces a 2×3 posterior grid figure for a selected event showing the prior density background and bEPIC posterior contours side-by-side for all priors.

---

## Dependencies

This repository is one part of a larger multi-project suite. It has **hard dependencies** on two sibling repositories that must be installed first:

### Required

**[bEPIC](https://github.com/danedwells/bEPIC)** — Bayesian earthquake early warning location algorithm.
Provides the core `EPIC_locate_prelim` module that this benchmark drives.

```bash
git clone git@github.com:danedwells/bEPIC.git bEPIC/
cd bEPIC
pip install -e .
```

**[priors](https://github.com/danedwells/nehrp_priors)** — Spatial seismic prior distributions.
Provides the `SeismicPrior` class (and `EtasPriorUpdater`) used to load, build, and cache `.tt3` prior files.

```bash
git clone git@github.com:danedwells/nehrp_priors.git priors/
cd priors
pip install -e .
```

`pycsep` is additionally required to build the Helmstetter prior (not auto-installed):

```bash
pip install git+https://github.com/SCECcode/pycsep.git
```

### Optional

**[etas_2](https://github.com/danedwells/etas_2)** — ETAS parameter inversion and catalog simulation.
Required only for the **time-dependent workflow** if you need to re-run the ETAS parameter inversion (`time_dependent_scripts/build_initial_prior.py`). Pre-inverted parameters (`data/california/etas_inversion/parameters_benchmark.json`) are committed to the repository, so you can run the dynamic benchmark without `etas_2` as long as that file is present.

---

## Installation

```bash
git clone git@github.com:danedwells/seismic_benchmark.git seismic_benchmark
cd seismic_benchmark
pip install -e .
```

This makes the `benchmark` package importable from the entry-point scripts.

---

## Directory structure

```
seismic_benchmark/
├── benchmark/                          # Python package — import as "benchmark"
│   ├── config.py                       # Prior filenames, bounds, benchmark/ETAS parameters (California)
│   ├── config_cascadia.py              # Same, for the Cascadia region
│   ├── runner.py                       # BenchmarkRunner class; run_prior / run_all_priors_parallel workers
│   ├── priors.py                       # build_and_cache_priors() — constructs .tt3 files from source data
│   ├── plots.py                        # Reusable figure helpers: maps, histograms, posterior grids
│   ├── usgs.py                         # USGS/IRIS download helpers; QuakeML parser; .run file builder
│   └── background.py                   # Background seismicity download/cache from USGS ComCat
├── preparation_scripts/                # Run these once before any benchmarking workflow
│   ├── build_priors.py                 # Construct all static prior .tt3 cache files
│   ├── case_study_preparation.py       # Download USGS catalogs + build .run files for all case studies
│   ├── build_station_availability.py   # Optional: per-event station-availability cache
│   └── ..._cascadia.py                 # Same steps, for the Cascadia region
├── time_independent_scripts/           # Static prior workflow
│   ├── run_benchmarks.py               # Main workflow: load priors → run bEPIC → plot
│   ├── case_studies.py                 # Case-study runner (preparation_scripts must run first)
│   └── run_cascadia.py                 # Same workflow for the Cascadia region
├── time_dependent_scripts/             # Dynamic ETAS prior workflow
│   ├── build_initial_prior.py          # ETAS parameter inversion — run once to regenerate parameters_benchmark.json
│   ├── run_benchmarks.py               # Main workflow with time-evolving ETAS prior
│   ├── case_studies.py                 # Case-study workflow with dynamic ETAS prior
│   └── run_cascadia.py                 # Same workflow for the Cascadia region
├── mixed_prior_scripts/                # Mixed (TI + ETAS) prior workflow
│   ├── run_benchmarks.py               # Main workflow: five TI × ETAS blended priors, serial event loop
│   └── case_studies.py                 # Case-study workflow with blended priors
├── plot_scripts/                       # Comparison plots that read results from more than one workflow
├── data_examination_scripts/           # Small standalone scripts for inspecting priors/catalogs/results
├── tests/                              # pytest unit tests (run with `pytest`)
│   ├── test_metrics.py                 # metrics.py — haversine, HDR levels, credible levels, coverage
│   ├── test_config.py                  # config.py — structure and value sanity checks
│   ├── test_runner.py                  # runner.py — DataFrame assembly, column normalisation, init
│   └── test_priors.py                  # priors.py — build_and_cache_priors error handling
├── data/                               # Input data — not committed to git
│   ├── california/                     # Main benchmark region
│   │   ├── run_files/                  # Per-event trigger sequences (*.run)
│   │   ├── etas_inversion/             # ETAS inversion outputs (parameters, catalog)
│   │   └── reference/                  # Reference catalog, background seismicity cache
│   ├── cascadia/                       # Same layout, for the Cascadia region
│   └── case_studies/                   # Per-case-study subdirs (run_files/, catalog cache)
│       ├── Ridgecrest/
│       ├── Ferndale/
│       ├── ElMayor/
│       └── MTJ_2024_M7/
├── results/                            # Generated outputs — not committed to git
│   ├── california/output/, california/figures/    # Same sub-layout as before (time_independent/time_dependent/mixed)
│   ├── cascadia/output/, cascadia/figures/
│   └── case_studies/{name}/output/, case_studies/{name}/figures/
├── pyproject.toml
├── README.md
└── CLAUDE.md                           # Developer notes
```

`data/` and `results/` are excluded from version control (see `.gitignore`). You will need to supply the trigger-sequence and reference-catalog files separately.

---

## Prior source data

These files are not included in this repository due to file size. Contact daniel.wells@usu.edu or danedwells@gmail.com if needed.

| Prior | Raw data source |
|-------|----------------|
| Gear1 | [GEAR1 global hazard table](https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model) |
| NSHM (gridded) | [gridded_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz) |
| NSHM (fault) | [fault_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/fault_moment_rates.xyz) |
| Helmstetter | [PyCSEP artifact](https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat) / [paper](https://hal.science/hal-00195399/document) |
| KDE_Seismicity | No external file needed — built from a downloaded USGS catalog. See `data_examination_scripts/examine_kde_prior.py`. |
| ETAS | Generated by `time_dependent_scripts/build_initial_prior.py` — see below |

Both NSHM files share the same 0.1° grid. Values are log₁₀-encoded moment rates (N·m/yr); `build_priors.py` exponentiates both, sums in linear space, then normalizes.

Place source files under `SeismicPrior.data_dir` in the paths specified by `benchmark/config.py` (`PRIOR_CONSTRUCTION_PARAMS['source_paths']`).

---

## Running the benchmark

### Preparation (run once before anything else)

#### Step 1 — Build prior cache (`preparation_scripts/build_priors.py`)

Constructs all static prior `.tt3` files from their raw source data and writes them to `SeismicPrior.data_dir`. Re-run whenever source data or construction parameters change.

```bash
python preparation_scripts/build_priors.py
```

#### Step 2 — Case study data (`preparation_scripts/case_study_preparation.py`)

Downloads USGS event catalogs and builds `.run` trigger files for all predefined case studies (Ridgecrest, Ferndale, El Mayor, MTJ_2024_M7). Must be run before any `case_studies.py` workflow script. Set `REDOWNLOAD=True` or `REBUILD_RUN_FILES=True` to force a refresh.

```bash
python preparation_scripts/case_study_preparation.py
```

---

### Time-independent workflow (static priors)

#### Standard benchmark (`time_independent_scripts/run_benchmarks.py`)

Evaluates bEPIC on a fixed catalog of pre-built `.run` trigger files. Written as a Jupyter-style script (cells delimited by `#%%`) — run cell-by-cell in an IDE or top-to-bottom as a plain script.

Two boolean flags near the top control which stages execute:

```python
RUN_ALL_PRIORS = False   # Run all five static priors in parallel
SKIP_RUN       = False   # Skip running bEPIC and load existing CSVs instead
```

Results appear in `results/california/output/time_independent/max_trigs_{N}/` and figures in `results/california/figures/time_independent/max_trigs_{N}/`.

**Figures produced:**

- `comparison_benchmark_locations.png` — all-prior map of final posterior locations
- `MTJ_grid_benchmark_locations.png` — 2×3 panel zoomed to the Mendocino Triple Junction, one prior per panel, with error lines from USGS to bEPIC location
- `MTJ_posterior_grid_{event_id}.png` — 2×3 panel for a single auto-selected MTJ event showing prior density (Blues) and posterior contours (Reds); set `MTJ_EVENT_ID` to pin a specific event
- Misfit and location error histograms for the full catalog and MTJ region

#### Case studies (`time_independent_scripts/case_studies.py`)

Runs bEPIC over a predefined aftershock sequence. **Requires `preparation_scripts/case_study_preparation.py` to have been run first** — catalog download and `.run` file construction are handled entirely there.

Set `ACTIVE_CASE_STUDY` to one of the predefined sequences:

| Key | Sequence |
|-----|----------|
| `Ridgecrest` | Ridgecrest 2019 aftershock sequence |
| `Ferndale` | Ferndale 2022 sequence |
| `ElMayor` | El Mayor-Cucapah 2010 aftershock sequence |
| `MTJ_2024_M7` | Mendocino Triple Junction M7 2024 sequence |

Two boolean flags control the run:

```python
RUN_ALL_PRIORS = True   # run all static priors in parallel
SKIP_RUN       = False  # load existing CSVs instead of running bEPIC
```

Results appear in `results/case_studies/{name}/output/` and figures in `results/case_studies/{name}/figures/`.

---

### Time-dependent workflow (dynamic ETAS prior)

The dynamic workflow maintains a running earthquake catalog and recomputes the ETAS conditional intensity λ(x, y, t) before each event is located, so the spatial prior captures the current aftershock distribution rather than long-term background rates.

#### Step 1 — ETAS parameter inversion (`time_dependent_scripts/build_initial_prior.py`)

**Skip this step if `data/california/etas_inversion/parameters_benchmark.json` already exists.**

Runs ETAS parameter inversion on a historical seismicity catalog using `etas_2`. Produces the pre-inverted parameter file consumed at runtime by `EtasPriorUpdater`. An inverted `parameters_benchmark.json` is included in this repository, produced by inverting ETAS on a downloaded catalog of California from 2000 to 2018, with a minimum magnitude of 3.

```bash
python time_dependent_scripts/build_initial_prior.py
```

This is slow (minutes to hours depending on catalog size). The resulting `parameters_benchmark.json` is committed so most users will not need to re-run this.

#### Step 2 — Dynamic benchmark (`time_dependent_scripts/run_benchmarks.py`)

Evaluates bEPIC on the same benchmark catalog, but with the ETAS prior updated in real time. Events are processed in **chronological order** so each location estimate sees only past seismicity.

Key flags:

```python
RUN_DYNAMIC_PRIORS    = False   # Run the dynamic ETAS benchmark
SKIP_RUN              = False   # Load existing CSVs instead of running bEPIC
DEBUG_PLOT_PRIOR      = False   # Plot the ETAS prior before each event (diagnostic)
ETAS_UPDATE_INTERVAL_S = 3600  # How often (in event time) to recompute the ETAS prior
PRIOR_ALPHA           = 0.5    # Tempering exponent — compresses ETAS dynamic range
```

The event loop works as follows:

1. `etas_update_fn(event_time)` — recomputes the ETAS conditional intensity at the current event time and calls `BenchmarkRunner.update_prior()` with the new `SeismicPrior`
2. bEPIC locates the event using the updated prior
3. `after_event_fn(event_id)` — appends the **USGS reference location** (not the bEPIC estimate) to the ETAS catalog so future priors reflect ground truth

Results go to `results/california/output/time_dependent/max_trigs_{N}/etas_dynamic_benchmark_results.csv` (nested one level deeper under a tag folder identifying the ETAS inversion settings used, so different settings don't overwrite each other).


A standalone single-event test block is also included — it builds a fresh ETAS prior for a configurable target event (`MTJ_EVENT_ID`) with a configurable lookback window of pre-event catalog entries, useful for diagnosing the prior for a specific event without running the full benchmark.

#### Step 3 — Case studies with dynamic ETAS (`time_dependent_scripts/case_studies.py`)

Same structure as the time-independent case studies, but uses the dynamic ETAS prior. Downloads the catalog, builds `.run` files, and runs bEPIC with the time-evolving prior.

---

### Mixed prior workflow (blended TI + ETAS)

Combines each of the five static priors with the dynamic ETAS prior using a weighted linear mixture evaluated at the ETAS grid resolution (0.1°):

```
combined = ALPHA * etas_prior + (1 - ALPHA) * ti_prior_resampled
```

The static prior is bilinearly resampled onto the ETAS grid before blending. `ALPHA = 0.5` by default and is set at the top of each script.

**Prerequisites**: both `preparation_scripts/build_priors.py` and `time_dependent_scripts/build_initial_prior.py` must have already been run.

#### Step 1 — Mixed-prior benchmark (`mixed_prior_scripts/run_benchmarks.py`)

Runs the same benchmark catalog as the other workflows, but the five blended priors all share one `EtasPriorUpdater` and must run serially (causal ETAS state).

Key flags:

```python
ALPHA                  = 0.5   # ETAS blend weight; (1-ALPHA) on the static prior
ETAS_UPDATE_INTERVAL_S = 0     # 0 = update ETAS before every event
RUN_MIXED              = True  # run the blended benchmark
SKIP_RUN               = False # load existing CSVs instead
DEBUG_PLOT_PRIOR       = False # plot raw ETAS grid before each update
```

The event loop evaluates the ETAS prior once per event, blends it with each of the five static priors, runs bEPIC five times, then appends the USGS reference location to the ETAS catalog before moving to the next event.

Results go to `results/california/output/mixed/max_trigs_{N}/{prior}_etas_mixed_benchmark_results.csv` (five files, one per TI prior). The same figure set as the other workflows is produced under `results/california/figures/mixed/max_trigs_{N}/`.

#### Step 2 — Mixed case studies (`mixed_prior_scripts/case_studies.py`)

Downloads a USGS catalog, builds `.run` files, then runs the blended-prior benchmark over the aftershock sequence. Case-study events (not the USGS reference catalog) are fed to the ETAS updater incrementally. Includes a standalone single-event section that builds fresh blended priors for a configurable focus event and plots location trajectories for all five mixed priors.

Set `ACTIVE_CASE_STUDY` to one of the same sequences (`Ridgecrest`, `Ferndale`, `ElMayor`, `MTJ_2024_M7`). Results appear in `results/case_studies/{name}/output/mixed/` and figures in `results/case_studies/{name}/figures/mixed/`.

---

### Cascadia region

The Pacific Northwest (Cascadia) region is supported alongside California, using the same three workflows above. Its preparation and benchmark scripts have the same names with `_cascadia` appended (e.g. `preparation_scripts/build_priors_cascadia.py`, `time_independent_scripts/run_cascadia.py`, `time_dependent_scripts/run_cascadia.py`), and its own config module, `benchmark/config_cascadia.py`. Data and results are kept separate under `data/cascadia/` and `results/cascadia/`.

---

### Running the unit tests

```bash
cd seismic_benchmark
pytest
```

Tests cover the pure-math and structural logic in the `benchmark` package — no network access or data files required. Requires `bEPIC` and `priors` to be installed.

---

## Output format

Each `{prior}_benchmark_results.csv` contains one row per (event, trigger version):

| Column | Description |
|--------|-------------|
| `event_id` | Event ID — matches the `.run` filename stem (an integer for the main California catalog, a USGS/ANSS string ID for case studies and Cascadia) |
| `version` / `n_trigs` | Trigger version / number of stations triggered so far |
| `posterior_lat`, `posterior_lon` | bEPIC posterior location |
| `best_misfit` | Best travel-time misfit |
| `best_like` | Best likelihood value |
| `best_prior` | Best prior value at posterior location |
| `frac_misfit` | Fractional travel-time error |
| `map_err_km` | Distance from the posterior location to the USGS catalog location |
| `posterior_confidence_level` | Calibration metric — the smallest credible region (0-1) that just contains the true USGS location. Values clustered near 0.5 indicate a well-calibrated posterior. |

A few more diagnostic columns (e.g. separate likelihood-only and prior-only versions of the columns above, and posterior coverage at fixed radii) are also written but are mainly used by the comparison plots in `plot_scripts/`.

---


## Key Decisions

### Ongoing
- Currently have retained priors/ as a separate repository, where all prior files live and a class to consistently read/write from them lives. This may be absorbed at a later date
- Currently have retained etas/ as a separate repository, where the code to perform both an ETAS inversion and evaluation live. This may be absorbed at a later date.
- Currently have bEPIC/ as a separate repository. This may be absorbed at a later date.

### Done
- Moved logic for catalog download and building run files into case_study_preparation.py. This file and build_priors.py now live in preparation_scripts/. These must be run before running benchmarking.
- Moved plotting scripts into plot_scripts/. These scripts examine both time_independent, time_dependent, and mixed benchmarking together, hence the logic for a separate script and folder.
- Separated the benchmarking scripts into 3 folders: time_independent_scripts, time_dependent_scripts, and mixed_prior_scripts. There is a lot of overlap between them. The reason for the separation is that time_dependent inherently must run differently - it is sequential, with the output each event affecting the prior for the next event.
- Included metric of posterior coverage (how much of the posterior probability mass is within fixed distances of 10, 25, 50, and 100 km of the true USGS location), usgs_credible level (what confidene contour of the posterior distribution of the final location does the true USGS location fall on? lower is better), and location error (great circle distance in km).
- Replaced the `Smooth_seismicity` prior with `KDE_Seismicity`, built directly from a downloaded USGS catalog instead of a separately-sourced file.
- Added a second region, Cascadia, alongside the original California benchmark, and a new case study (`MTJ_2024_M7`). `data/` and `results/` are now split into `california/` and `cascadia/` subfolders to keep the two regions separate.


---


## Known limitations / work in progress

- Waveform processing is not currently used and will be implemented in the near future for magnitude calculation.
- Magnitude Calculation on a per station basis by will be implemented in the future. This will entail measuring peak displacement from .mseed files and calculating magnitude on a per-station basis. This calcluation will happen AFTER event location (not part of a simultaneous inversion).
- The `REFERENCE` workflow (high-resolution reference locations) is currently disabled.
- The benchmark_runner API may change without notice in future iterations as needed to accomplish research goals.
- Streamlining of dependencies, and how they interact with this repository, will be done at some unknown point in the future.
- There is a minimal formal test suite - basic mathematics and functionality are tested within tests/
- Interfacing with other repositories for further benchmarking, inclusion of AI/ML models, and evaluation in real-time is expected in the future.
