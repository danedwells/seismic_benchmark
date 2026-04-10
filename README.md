# seismic_benchmark

> **Work in progress.** This repository is under active development. APIs, file layouts, and workflows may change without notice. This repository is being developed with the assistance of LLM/AI tools.

A benchmarking framework for evaluating the [bEPIC](https://github.com/danedwells/bEPIC) Bayesian earthquake early warning location algorithm across different spatial prior distributions. Given a set of real earthquake trigger sequences, it runs bEPIC iteratively as station triggers arrive and compares the resulting posterior locations against USGS ANSS catalog reference positions.

---

## What it does

For each event in the test catalog, bEPIC is run once per trigger version (i.e., once each time a new station triggers), simulating real-time location updates. This is repeated for six different spatial priors:

| Prior | Source |
|-------|--------|
| `Gear1` | GEAR1 global seismic hazard table |
| `NSHM` | USGS NSHM gridded + fault moment rates (summed in linear space) |
| `Helmstetter` | Helmstetter (2007) smoothed seismicity (requires pycsep) |
| `Smooth_seismicity` | Pre-built US/Canada smoothed seismicity grid |
| `ETAS` | ETAS-derived spatial intensity (time-dependent; optional) |
| `Uniform` | Uninformative baseline; equivalent to having no prior. |

Results (posterior lat/lon, travel-time misfits, location errors vs. ANSS) are written to CSV and visualized as maps and histograms. Each run also produces a 2×3 posterior grid figure for a selected event showing the prior density background and bEPIC posterior contours side-by-side for all six priors.

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
Provides the `SeismicPrior` class used to load, build, and cache `.tt3` prior files.

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
Can be used to generate a time-dependent ETAS spatial prior (a `.tt3` file) for use as the `ETAS` prior in benchmarks. If you don't need the ETAS prior, this dependency can be ignored.

---

## Installation


```bash
mkdir seismic_benchmark
git clone git@github.com:danedwells/seismic_benchmark.git seismic_benchmark
cd seismic_benchmark
pip install -e .
```

This makes the `benchmark` package importable from the `scripts/` entry points.

---

## Directory structure

```
seismic_benchmark/
├── benchmark/                  # Python package — import as "benchmark"
│   ├── config.py               # Prior filenames, bounds, benchmark parameters
│   ├── runner.py               # BenchmarkRunner class; run_prior / run_all_priors_parallel workers
│   ├── priors.py               # build_and_cache_priors() — constructs .tt3 files from source data
│   ├── plots.py                # plot_prior_histograms(), plot_posterior_grid() — reusable figure helpers
│   ├── usgs.py                 # USGS/IRIS download helpers; QuakeML parser; .run file builder
│   └── background.py           # Background seismicity download/cache from USGS ComCat
├── scripts/                    # Entry-point scripts (run these directly)
│   ├── build_priors.py         # One-time prior construction — run before anything else
│   ├── run_benchmarks.py       # Main workflow: load priors → run bEPIC → plot
│   ├── case_studies.py         # Case-study workflow: download catalog → build .run files → run bEPIC → plot
│   └── examine_catalog.py      # Catalog QC: maps, magnitude-time, USGS verification
├── data/                       # Input data — not committed to git
│   ├── run_files/              # Per-event trigger sequences (*.run) for the standard benchmark
│   ├── case_studies/           # Per-case-study subdirs (run_files/, catalog cache)
│   │   ├── Ridgecrest/
│   │   ├── Ferndale/
│   │   └── ElMayor/
│   └── reference/              # Reference catalog, background seismicity cache
├── results/                    # Generated outputs — not committed to git
│   ├── output/max_trigs_N/     # Benchmark CSVs: {prior}_benchmark_results.csv
│   ├── figures/max_trigs_N/    # Comparison maps and histograms
│   └── case_studies/           # Per-case-study output and figures
├── pyproject.toml
├── README.md
└── CLAUDE.md                   # Developer notes and planned API direction
```

`data/` and `results/` are excluded from version control (see `.gitignore`). You will need to supply the `data/run_files/` trigger sequences and `data/reference/bEPIC_testing_catalog.txt` separately. Later versions of this repository may include the raw data files, which take significant space.

## Prior source data

These files are not included in this repository due to file size. Contact daniel.wells@usu.edu or danedwells@gmail.com if needed.

| Prior | Raw data source |
|-------|----------------|
| Gear1 | [GEAR1 global hazard table](https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model) |
| NSHM (gridded) | [gridded_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz) |
| NSHM (fault) | [fault_moment_rates.xyz](https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/fault_moment_rates.xyz) |
| Helmstetter | [PyCSEP artifact](https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat) / [paper](https://hal.science/hal-00195399/document) |
| Smooth_seismicity | Williamson smoothed seismicity — contact Amy Williamson (Amy.Williamson@berkeley.edu) |
| ETAS | Generated by `etas_2` — see [etas_2 repo](https://github.com/danedwells/etas_2) |

Both NSHM files (`gridded_moment_rates.xyz` and `fault_moment_rates.xyz`) share the same 0.1° grid. Values are log₁₀-encoded moment rates (N·m/yr); `build_priors.py` exponentiates both, sums them in linear space, then normalizes.

Place source files under `SeismicPrior.data_dir` in the paths specified by `benchmark/config.py` (`PRIOR_CONSTRUCTION_PARAMS['source_paths']`).

## Existing data

These files are not currently included in this repository due to file size and may be included in future iterations. Please contact daniel.wells@usu.edu or danedwells@gmail.com if needed.

#### bEPIC Run Files

    bEPIC run files ({event_ID}.run) are required to run the standard benchmark. These files are not currently included in this repository and may be included in future iterations.

    For case studies, .run files are generated automatically from USGS phase data by scripts/case_studies.py — no pre-built files are needed.

#### Reference catalog locations

    bEPIC_testing_catalog.txt — an example catalog connecting .run files to USGS/ANSS event IDs.

    background_seismicity.parquet — a downloadable catalog of USGS events in the California region. Downloaded automatically on first run.


---

## Running the benchmark

### Step 0 — Build prior cache (`scripts/build_priors.py`)

**Run this once before anything else.** Constructs all prior `.tt3` files from their raw source data and writes them to `SeismicPrior.data_dir`. Re-run whenever source data or construction parameters change.

```bash
cd seismic_benchmark
python scripts/build_priors.py
```

This is a prerequisite for both `run_benchmarks.py` and `case_studies.py`.

---

### Standard benchmark (`scripts/run_benchmarks.py`)

Evaluates bEPIC on a fixed catalog of pre-built `.run` trigger files. Written as a Jupyter-style script (cells delimited by `#%%`) — run cell-by-cell in an IDE or top-to-bottom as a plain script.

Two boolean flags near the top control which stages execute:

```python
REFERENCE      = False   # Run high-resolution reference locations (currently unused)
RUN_ALL_PRIORS = False   # Run all six priors in parallel
```

**Typical run:**

1. Set `RUN_ALL_PRIORS = True` to run bEPIC across all six priors in parallel.
2. Results appear in `results/output/max_trigs_{N}/` and figures in `results/figures/max_trigs_{N}/`.

**Figures produced:**

- `comparison_benchmark_locations.png` — all-prior map of final posterior locations
- `MTJ_grid_benchmark_locations.png` — 2×3 panel zoomed to the Mendocino Triple Junction, one prior per panel, with error lines from USGS to bEPIC location
- `MTJ_posterior_grid_{event_id}.png` — 2×3 panel for a single auto-selected MTJ event showing prior density (Blues) and posterior contours (Reds); set `MTJ_EVENT_ID` to pin a specific event
- Misfit and location error histograms for both the full catalog and MTJ region

---

### Case studies (`scripts/case_studies.py`)

Runs bEPIC on a user-defined earthquake sequence (e.g. an aftershock sequence) downloaded live from USGS. Unlike the standard benchmark, there are no pre-built `.run` files — they are constructed on the fly from USGS phase data.

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
RUN_ALL_PRIORS   = False   # Run all six priors in parallel
```

**Typical first run:**

1. Set `DOWNLOAD_CATALOG = True` and `BUILD_RUN_FILES = True` to fetch the catalog and phase arrivals from USGS and IRIS. The catalog is cached as a parquet file; `.run` files are written to `data/case_studies/{name}/run_files/`. Subsequent runs can leave both flags `False` to reuse cached data.
2. Set `RUN_ALL_PRIORS = True` to run bEPIC across all priors.
3. Results appear in `results/case_studies/{name}/output/` and figures in `results/case_studies/{name}/figures/`.

**Figures produced** (same structure as standard benchmark, scoped to the case study region):

- Overall location map comparing all priors
- 2×3 grid of per-prior location maps with error lines
- 2×3 posterior grid figure for a configurable single event (`FOCUS_EVENT_ID`)
- Location error and fractional misfit histograms

The `build_run_files` step queries USGS ComCat for phase picks and IRIS FDSNWS for station coordinates. It applies a rate-limiting delay between events (default 1.5 s) and skips events that already have a `.run` file. Phase data is read from `phases.csv` if available; otherwise parsed from `quakeml.xml`.

---

### Catalog examination

```bash
python scripts/examine_catalog.py
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

- The `ETAS` prior requires an externally generated `.tt3` file from `etas_2`; there is no automated pipeline connecting them yet.
- The `REFERENCE` workflow (high-resolution reference locations) is currently disabled.
- There is no formal test suite; validation is done by visual inspection of output figures.
- The `BenchmarkRunner` API is expected to change.
- Interfacing with other repositories for further benchmarking, inclusion of AI/ML models, and evaluation of data in real-time is expected in the future.
