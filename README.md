# seismic_benchmark

> **Work in progress.** This repository is under active development. APIs, file layouts, and workflows may change without notice. This repository is being developed with the assistance of LLM/AI tools.

A benchmarking framework for evaluating the [bEPIC](https://github.com/danedwells/bEPIC) Bayesian earthquake early warning location algorithm across different spatial prior distributions. Given a set of real earthquake trigger sequences, it runs bEPIC iteratively as station triggers arrive and compares the resulting posterior locations against USGS ANSS catalog reference positions.

Two parallel workflows are provided:

- **Time-independent** (`time_independent_scripts/`) — five static spatial priors, unchanged per-event, evaluated in parallel.
- **Time-dependent** (`time_dependent_scripts/`) — a dynamic ETAS prior that evolves as events are located, capturing the time-varying spatial distribution of aftershock hazard.

---

## What it does

For each event in the test catalog, bEPIC is run once per trigger version (i.e., once each time a new station triggers), simulating real-time location updates. This is repeated across spatial priors:

| Prior | Source | Workflow |
|-------|--------|----------|
| `Gear1` | GEAR1 global seismic hazard table | Time-independent |
| `NSHM` | USGS NSHM gridded + fault moment rates (summed in linear space) | Time-independent |
| `Helmstetter` | Helmstetter (2007) smoothed seismicity (requires pycsep) | Time-independent |
| `Smooth_seismicity` | Pre-built US/Canada smoothed seismicity grid | Time-independent |
| `Uniform` | Uninformative baseline; equivalent to having no prior | Time-independent |
| `ETAS` | Dynamic ETAS conditional intensity λ(x, y, t), updated before each event | Time-dependent |

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
Required only for the **time-dependent workflow** if you need to re-run the ETAS parameter inversion (`time_dependent_scripts/build_initial_prior.py`). Pre-inverted parameters (`data/etas_inversion/parameters_benchmark.json`) are committed to the repository, so you can run the dynamic benchmark without `etas_2` as long as that file is present.

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
│   ├── config.py                       # Prior filenames, bounds, benchmark/ETAS parameters
│   ├── runner.py                       # BenchmarkRunner class; run_prior / run_all_priors_parallel workers
│   ├── priors.py                       # build_and_cache_priors() — constructs .tt3 files from source data
│   ├── plots.py                        # Reusable figure helpers: maps, histograms, posterior grids
│   ├── usgs.py                         # USGS/IRIS download helpers; QuakeML parser; .run file builder
│   └── background.py                   # Background seismicity download/cache from USGS ComCat
├── time_independent_scripts/           # Static prior workflow (run these for the standard benchmark)
│   ├── build_priors.py                 # One-time prior construction — run before anything else
│   ├── run_benchmarks.py               # Main workflow: load priors → run bEPIC → plot
│   ├── case_studies.py                 # Case-study workflow: download catalog → build .run files → run → plot
│   ├── examine_catalog.py              # Catalog QC: maps, magnitude-time, USGS verification
│   └── test_ss_prior.py                # Small diagnostic for the smooth_seismicity prior
├── time_dependent_scripts/             # Dynamic ETAS prior workflow (new)
│   ├── build_initial_prior.py          # ETAS parameter inversion — run once to regenerate parameters_benchmark.json
│   ├── run_benchmarks.py               # Main workflow with time-evolving ETAS prior
│   ├── case_studies.py                 # Case-study workflow with dynamic ETAS prior
│   └── examine_catalog.py              # Catalog QC (same as time-independent version)
├── data/                               # Input data — not committed to git
│   ├── run_files/                      # Per-event trigger sequences (*.run) for the standard benchmark
│   ├── etas_inversion/                 # ETAS inversion outputs
│   │   ├── parameters_benchmark.json   # Pre-inverted ETAS parameters (committed)
│   │   ├── catalog_benchmark.csv       # Catalog used for inversion
│   │   └── input/downloaded_catalog.csv
│   ├── case_studies/                   # Per-case-study subdirs (run_files/, catalog cache)
│   │   ├── Ridgecrest/
│   │   ├── Ferndale/
│   │   └── ElMayor/
│   └── reference/                      # Reference catalog, background seismicity cache
├── results/                            # Generated outputs — not committed to git
│   ├── output/
│   │   ├── max_trigs_{N}/              # Static prior CSVs: {prior}_benchmark_results.csv
│   │   └── time_dependent/max_trigs_{N}/  # Dynamic ETAS CSV
│   ├── figures/
│   │   ├── max_trigs_{N}/              # Maps, histograms, posterior grids (static priors)
│   │   └── time_dependent/max_trigs_{N}/  # Same figures for dynamic ETAS
│   └── case_studies/                   # Per-case-study output and figures
├── pyproject.toml
├── README.md
└── CLAUDE.md                           # Developer notes
```

`data/` and `results/` are excluded from version control (see `.gitignore`). You will need to supply the `data/run_files/` trigger sequences and `data/reference/bEPIC_testing_catalog.txt` separately.

---

## Prior source data

These files are not included in this repository due to file size. Contact daniel.wells@usu.edu or danedwells@gmail.com if needed.

| Prior | Raw data source |
|-------|----------------|
| Gear1 | [GEAR1 global hazard table](https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model) |
| NSHM (gridded) | [gridded_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz) |
| NSHM (fault) | [fault_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/fault_moment_rates.xyz) |
| Helmstetter | [PyCSEP artifact](https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat) / [paper](https://hal.science/hal-00195399/document) |
| Smooth_seismicity | Williamson smoothed seismicity — contact Amy Williamson (Amy.Williamson@berkeley.edu) |
| ETAS | Generated by `time_dependent_scripts/build_initial_prior.py` — see below |

Both NSHM files share the same 0.1° grid. Values are log₁₀-encoded moment rates (N·m/yr); `build_priors.py` exponentiates both, sums in linear space, then normalizes.

Place source files under `SeismicPrior.data_dir` in the paths specified by `benchmark/config.py` (`PRIOR_CONSTRUCTION_PARAMS['source_paths']`).

---

## Running the benchmark

### Time-independent workflow (static priors)

#### Step 1 — Build prior cache (`time_independent_scripts/build_priors.py`)

**Run this once before anything else.** Constructs all static prior `.tt3` files from their raw source data and writes them to `SeismicPrior.data_dir`. Re-run whenever source data or construction parameters change.

```bash
python time_independent_scripts/build_priors.py
```

#### Step 2 — Standard benchmark (`time_independent_scripts/run_benchmarks.py`)

Evaluates bEPIC on a fixed catalog of pre-built `.run` trigger files. Written as a Jupyter-style script (cells delimited by `#%%`) — run cell-by-cell in an IDE or top-to-bottom as a plain script.

Two boolean flags near the top control which stages execute:

```python
RUN_ALL_PRIORS = False   # Run all five static priors in parallel
SKIP_RUN       = False   # Skip running bEPIC and load existing CSVs instead
```

Results appear in `results/output/max_trigs_{N}/` and figures in `results/figures/max_trigs_{N}/`.

**Figures produced:**

- `comparison_benchmark_locations.png` — all-prior map of final posterior locations
- `MTJ_grid_benchmark_locations.png` — 2×3 panel zoomed to the Mendocino Triple Junction, one prior per panel, with error lines from USGS to bEPIC location
- `MTJ_posterior_grid_{event_id}.png` — 2×3 panel for a single auto-selected MTJ event showing prior density (Blues) and posterior contours (Reds); set `MTJ_EVENT_ID` to pin a specific event
- Misfit and location error histograms for the full catalog and MTJ region

#### Step 3 — Case studies (`time_independent_scripts/case_studies.py`)

Runs bEPIC on a user-defined aftershock sequence downloaded live from USGS. Unlike the standard benchmark, there are no pre-built `.run` files — they are constructed on the fly from USGS phase data.

Set `ACTIVE_CASE_STUDY` to one of the predefined sequences:

| Key | Sequence |
|-----|----------|
| `Ridgecrest` | Ridgecrest 2019 aftershock sequence |
| `Ferndale` | Ferndale 2022 sequence |
| `ElMayor` | El Mayor-Cucapah 2010 aftershock sequence |

Three boolean flags control the stages:

```python
DOWNLOAD_CATALOG = False   # Re-download the USGS event catalog (else use cache)
BUILD_RUN_FILES  = False   # Fetch USGS phase data and build .run trigger files
RUN_ALL_PRIORS   = False   # Run all static priors in parallel
```

Results appear in `results/case_studies/{name}/output/` and figures in `results/case_studies/{name}/figures/`.

---

### Time-dependent workflow (dynamic ETAS prior)

The dynamic workflow maintains a running earthquake catalog and recomputes the ETAS conditional intensity λ(x, y, t) before each event is located, so the spatial prior captures the current aftershock distribution rather than long-term background rates.

#### Step 1 — ETAS parameter inversion (`time_dependent_scripts/build_initial_prior.py`)

**Skip this step if `data/etas_inversion/parameters_benchmark.json` already exists.**

Runs ETAS parameter inversion on a historical seismicity catalog using `etas_2`. Produces the pre-inverted parameter file consumed at runtime by `EtasPriorUpdater`. There is an inverted parameters_benchmark.josn included in this repository. This was produced by inverting ETAS on a downloaded catalog of California from 2000 to 2018, with a minimum magnitude of 3.

```bash
python time_dependent_scripts/build_initial_prior.py
```

This is slow (minutes to hours depending on catalog size). The resulting `parameters_benchmark.json` is committed so most users will not need to re-run this.

#### Step 2 — Dynamic benchmark (`time_dependent_scripts/run_benchmarks.py`)

Evaluates bEPIC on a 740-event catalog, but with the ETAS prior updated in real time. Events are processed in **chronological order** so each location estimate sees only past seismicity.

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

Results go to `results/output/time_dependent/max_trigs_{N}/etas_dynamic_benchmark_results.csv`.


A standalone single-event test block is also included — it builds a fresh ETAS prior for a configurable target event (`MTJ_EVENT_ID`) with a configurable lookback window of pre-event catalog entries, useful for diagnosing the prior for a specific event without running the full benchmark.

#### Step 3 — Case studies with dynamic ETAS (`time_dependent_scripts/case_studies.py`)

Same structure as the time-independent case studies, but uses the dynamic ETAS prior. Downloads the catalog, builds `.run` files, and runs bEPIC with the time-evolving prior.

---

### Catalog examination

```bash
python time_independent_scripts/examine_catalog.py
# or
python time_dependent_scripts/examine_catalog.py
```

Produces maps and a USGS online verification report for the standard test catalog. Outputs are saved to `data/reference/`.

---

## Output format

Each `{prior}_benchmark_results.csv` contains one row per (event, trigger version):

| Column | Description |
|--------|-------------|
| `event_id` | Integer event ID (matches `.run` filename stem) |
| `version` | Trigger version (increments as each new station triggers) |
| `posterior_lat`, `posterior_lon` | bEPIC posterior location |
| `best_misfit` | Best travel-time misfit |
| `best_like` | Best likelihood value |
| `best_prior` | Best prior value at posterior location |
| `frac_misfit` | Fractional travel-time error |
| `location_error_km` | Distance to USGS catalog location (added post-hoc) |

---

## Known limitations / work in progress

- Current top priority - map the posterior mass of a final location from bEPIC to the distance to the 'true' (USGS) location to get a sense of how well the posterior is capturing true uncertainty. This will be implemented in both time_dependent and time_independent scripts.
- The `REFERENCE` workflow (high-resolution reference locations) is currently disabled.
- The benchmark_runner API may change without notice in future iterations as needed to accomplish research goals.
- Streamlining of dependencies, and how they interact with this repository, will be done at some unknown point in the future.
- There is no formal test suite; validation is done by visual inspection of output figures.
- Interfacing with other repositories for further benchmarking, inclusion of AI/ML models, and evaluation in real-time is expected in the future.
