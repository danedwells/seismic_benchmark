# seismic_benchmark

> **Work in progress.** This repository is under active development. APIs, file layouts, and workflows may change without notice. This repository is being developed with the assistance of LLM/AI tools.

A benchmarking framework for evaluating the [bEPIC](https://github.com/danedwells/bEPIC) Bayesian earthquake early warning location algorithm across different spatial prior distributions. Given a set of real earthquake trigger sequences, it runs bEPIC iteratively as station triggers arrive and compares the resulting posterior locations against USGS ANSS catalog reference positions.

---

## What it does

For each event in the test catalog, bEPIC is run once per trigger version (i.e., once each time a new station triggers), simulating real-time location updates. This is repeated for six different spatial priors:

| Prior | Source |
|-------|--------|
| `Gear1` | GEAR1 global seismic hazard table |
| `NSHM` | USGS National Seismic Hazard Model gridded moment rates |
| `Helmstetter` | Helmstetter (2007) smoothed seismicity (requires pycsep) |
| `Smooth_seismicity` | Pre-built US/Canada smoothed seismicity grid |
| `ETAS` | ETAS-derived spatial intensity (time-dependent; optional) |
| `Uniform` | Uninformative baseline; equivalent to having no prior. |

Results (posterior lat/lon, travel-time misfits, location errors vs. ANSS) are written to CSV and visualized as maps and histograms.

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
│   ├── runner.py               # BenchmarkRunner class; USGS/IRIS retrieval; run_prior worker
│   ├── priors.py               # build_and_cache_priors() — constructs .tt3 files
│   └── background.py           # Background seismicity download/cache from USGS ComCat
├── scripts/                    # Entry-point scripts (run these directly)
│   ├── run_benchmarks.py       # Main workflow: load priors → run bEPIC → plot
│   └── examine_catalog.py      # Catalog QC: maps, magnitude-time, USGS verification
├── data/                       # Input data — not committed to git
│   ├── run_files/              # Per-event trigger sequences (*.run)
│   └── reference/              # Reference catalog, background seismicity cache
├── results/                    # Generated outputs — not committed to git
│   ├── output/max_trigs_N/     # Benchmark CSVs: {prior}_benchmark_results.csv
│   └── figures/max_trigs_N/    # Comparison maps and histograms
├── pyproject.toml
├── README.md
└── CLAUDE.md                   # Developer notes and planned API direction
```

`data/` and `results/` are excluded from version control (see `.gitignore`). You will need to supply the `data/run_files/` trigger sequences and `data/reference/bEPIC_testing_catalog.txt` separately. Later versions of this repository may include the raw data files, which take significant space.

## Existing Priors

These are not included in this repository due to file size. This may change in future iterations. Please contact daniel.wells@usu.edu or danedwells@gmail.com if these are needed.

#### Existing Priors Links and Info For Raw Data


    • Gear1 () - https://pubs.geoscienceworld.org/ssa/bssa/article/105/5/2538/332070/GEAR1-A-Global-Earthquake-Activity-Rate-Model
    • NSHM () - https://data.opensha.org/nshm23/reports/branch_averaged_gridded/resources/gridded_moment_rates.xyz
    or https://data.opensha.org/nshm23/reports/branch_averaged_gridded/#regional-nucleation-rates
    • Helmstetter (2007) – from PyCSEP package https://github.com/cseptesting/pycsep/blob/main/csep/artifacts/ExampleForecasts/GriddedForecasts/helmstetter_et_al.hkj-fromXML.dat
        AND from paper itself https://hal.science/hal-00195399/document
    • Smoothed Seismicity (Williamson) – Google Drive https://drive.google.com/drive/folders/1qjJDD1CV43Afhp0xP7-Z9g4VJY8RIvYZ and/or email from Amy Williamson @ Amy.Williamson@berkeley.edu
    • ETAS_2 output – Running on example catalog, see ETAS_2 github (forked from ETAS) https://github.com/danedwells/etas_2

## Existing data

These files are not currently included in this repository due to file size and may be included in future iterations. Please contact daniel.wells@usu.edu or danedwells@gmail.com if these are needed.

#### bEPIC Run Files

    bEPIC run files ({event_ID}.run) are required to run the bEPIC algorithm. These files are not currently included in this repository and may be included in future iterations. 

#### Reference catalog locations

    bEPIC_testing_catalog.txt - an example catalog connecting .run files to USGS/ANSS event ids

    background_seisimicity.parquet - a downloadable catalog of USGS events in the California region. This catalog is downloadable with the code in this repository.


---

## Running the benchmark

The main entry point is `scripts/run_benchmarks.py`. It is written as a Jupyter-style script (cells delimited by `#%%`) and can be run cell-by-cell in an IDE or top-to-bottom as a plain script.

Three boolean flags near the top control which stages execute:

```python
construct      = False   # Rebuild all prior .tt3 files from source data
REFERENCE      = False   # Run high-resolution reference locations (unused)
RUN_ALL_PRIORS = False   # Run all six priors in parallel
```

**Typical first run:**

1. Set `construct = True` to build and cache the `.tt3` prior files (requires source data in `SeismicPrior.data_dir`).
2. Set `RUN_ALL_PRIORS = True` to run bEPIC across all priors in parallel.
3. Results appear in `results/output/max_trigs_{N}/` and figures in `results/figures/max_trigs_{N}/`.

**Catalog examination:**

```bash
python scripts/examine_catalog.py
```

Produces maps and a USGS online verification report for the test catalog. Outputs are saved to `data/reference/`.

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
