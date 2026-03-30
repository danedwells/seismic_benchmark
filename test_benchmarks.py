#%%
# Get priors - separate repo
from priors import SeismicPrior

# Get bEPIC - separate repo
from bEPIC import EPIC_locate_prelim

# Standard imports
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Get custom class for this comparison
import benchmark_runner

# Get root. Priors are in a different repo (priors/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get directory of THIS project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

data_dir = SeismicPrior.data_dir  # priors/data/

# Cached .tt3 paths for each prior
cache_paths = {
    'Gear1':             os.path.join(data_dir, 'GEAR1_prior.tt3'),
    'NSHM':              os.path.join(data_dir, 'USGS_NSHM_prior.tt3'),
    'Helmstetter':       os.path.join(data_dir, 'helmstetter_prior.tt3'),
    'Smooth_seismicity': os.path.join(data_dir, 'prior_seis_grid_US_Canada.tt3'),
    'ETAS':              os.path.join(data_dir, 'etas_prior_20080101_000000.tt3'),  # set filename as needed
    'Uniform':           None,   # no prior — use_prior=False, uniform weighting
}

#%%
"""
# --- Step 1: Build and cache all priors ---
# Set construct = True to rebuild; False to skip if .tt3 files already exist.
"""
construct = True

if construct:
    try:
        p = SeismicPrior.from_gear1(os.path.join(data_dir, 'GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'))
        p.to_tt3(cache_paths['Gear1'])
        print("Gear1: built and cached.")
    except Exception as e:
        print(f"Gear1: failed — {e}")

    try:
        p = SeismicPrior.from_nshm(os.path.join(data_dir, 'USGS_NSHM_data', 'gridded_moment_rates.xyz'))
        p.to_tt3(cache_paths['NSHM'])
        print("NSHM: built and cached.")
    except Exception as e:
        print(f"NSHM: failed — {e}")

    try:
        p = SeismicPrior.from_helmstetter()
        p.to_tt3(cache_paths['Helmstetter'])
        print("Helmstetter: built and cached.")
    except Exception as e:
        print(f"Helmstetter: failed — {e}")

    try:
        SeismicPrior.from_smooth_seismicity()  # validates the file loads cleanly
        print("Smooth_seismicity: ready (pre-built).")
    except Exception as e:
        print(f"Smooth_seismicity: failed — {e}")

    # TODO - incorporate calling this to build it, or have a dataframe or dict of saved options? 
    # Lots of ways to call this.
    # ETAS requires external inputs — build separately and save:
    #   p = SeismicPrior.from_etas(lats, lons, lambda_grid, forecast_time=t, metadata={...})
    #   p.to_tt3(cache_paths['ETAS'])
    print("ETAS: skipped (requires external ETAS output — build manually).")


#%%

"""
# --- Step 2: Select and load prior for bEPIC ---
# Options: 'Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform'

Single event - but with the imported benchmark_runner
classes and modules
"""
# Get prior
selected = 'smooth_seismicity' # NOT sensitive to upper or lowercase (GEAR1 and gear1 both work)
cache_key = next(k for k in cache_paths if k.lower() == selected.lower())
if cache_paths[cache_key] is None:
    p = SeismicPrior.from_tt3(cache_paths['NSHM'])  # geometry placeholder only
    params_use_prior = False
else:
    p = SeismicPrior.from_tt3(cache_paths[cache_key])
    params_use_prior = True

# Pass prior to param
params = EPIC_locate_prelim.EPIC_PARAMS()
params.prior = p
params.use_prior = params_use_prior
params.GridSize = 50 # Number of grid points or slices
params.GridKm = 100 # Total size of grid (in km, one dimension) # Grid spacing = gridkm/gridsize
params.method = 'EPIC C'  
params.MAX_EVENT_TRIGS = 25 # Maximum amount of triggers to use


# Create directory and benchmark runner object
run_dir = os.path.join(PROJECT_ROOT, 'run_files')
runner = benchmark_runner.BenchmarkRunner(prior=p, params=params, run_dir=run_dir)

# RUN A SINGLE EVENT
#event_id = 126625
#runner.run_event(event_id)
#results = runner.results

# Run a single benchmark OR create reference locations

event_ids = sorted(
    int(f.stem) # Extract everything before the period
    # A str.split(.)[0] could also work here.
    for f in (Path(run_dir)).glob('*.run')
)
selected = 'smooth_seismicity'

REFERENCE = True
REF_FILE_NAME = f"REFERENCE_{params.MAX_EVENT_TRIGS}.csv"
RERUN = False
csv_path = os.path.join(PROJECT_ROOT, f'output/{selected}_benchmark_results.csv')
if REFERENCE == True:
    
    csv_path = os.path.join(PROJECT_ROOT, f'reference_locations/{REF_FILE_NAME}')

if RERUN:
    runner.run_all(event_ids)
    os.makedirs(os.path.join(PROJECT_ROOT, 'output/'), exist_ok=True)
    pd.DataFrame([
        {'event_id': eid, 'version': ver, 'posterior_lat': t.posterior_lat,
         'posterior_lon': t.posterior_lon, 'best_misfit': t.best_misfit,
         'best_like': t.best_like, 'best_prior': t.best_prior}
        for (eid, ver), (t, _) in runner.results.items()
    ]).sort_values(['event_id', 'version']).to_csv(csv_path, index=False)


#%%
# --- Run all priors in parallel ---
from concurrent.futures import ProcessPoolExecutor, as_completed

RUN_ALL_PRIORS = False
GRID_SIZE_FULL = 50
GRID_KM_FULL   = 100
MAX_TRIGS = 25

if RUN_ALL_PRIORS:
    job_args = [
        {
            'prior_name': name,
            'cache_path': path,
            'nshm_path':  cache_paths['NSHM'],  # geometry fallback for Uniform
            'run_dir':    run_dir,
            'output_dir': os.path.join(PROJECT_ROOT, 'output'),
            'grid_size':  GRID_SIZE_FULL,
            'grid_km':    GRID_KM_FULL,
            'max_trigs':  MAX_TRIGS,
        }
        for name, path in cache_paths.items()
    ]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(benchmark_runner.run_prior, args): args['prior_name']
                   for args in job_args}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"{name}: done")
            except Exception as e:
                print(f"{name}: FAILED — {e}")

#%%


#%%
# --- Load reference locations (static; smooth_seismicity run to completion) ---
ref_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'reference_locations', REF_FILE_NAME))
ref_final = ref_df.groupby('event_id').last().reset_index()

def get_unique_stations(run_dir):
    """Return a DataFrame of unique stations (by station+network) across all run files."""
    frames = [pd.read_csv(f, usecols=['station', 'network', 'longitude', 'latitude'])
              for f in Path(run_dir).glob('*.run')]
    return pd.concat(frames).drop_duplicates(subset=['station', 'network']).reset_index(drop=True)

stations_df = get_unique_stations(run_dir)

#%%
# --- Plot with cartopy ---
results_df = pd.read_csv(csv_path)
final_df = results_df.groupby('event_id').last().reset_index()
final_lats = final_df['posterior_lat'].values
final_lons = final_df['posterior_lon'].values
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})

ax.set_extent([-126, -113, 31, 44], crs=proj)   # California + neighbors
ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

ax.scatter(ref_final['posterior_lon'], ref_final['posterior_lat'],
           s=10, color='black', alpha=0.4, transform=proj,
           label='Reference (smooth_seismicity)', zorder=1)
ax.scatter(final_lons, final_lats,
           s=8, color='crimson', alpha=0.6, transform=proj,
           label=f'bEPIC estimates (n={len(final_lats)})', zorder=2)
ax.scatter(stations_df['longitude'], stations_df['latitude'],
           s=20, color='orange', edgecolor='k', alpha=0.7, marker='v', transform=proj,
           label='Stations', zorder=0)

ax.set_title(f'bEPIC final locations — prior: {selected}')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
os.makedirs(os.path.join(PROJECT_ROOT, 'figures/'), exist_ok=True)
plt.savefig(os.path.join(PROJECT_ROOT, f'figures/{selected}_benchmark_locations.png'), dpi=150)
#plt.savefig(os.path.join(PROJECT_ROOT, f'figures/REFERENCE_benchmark_locations.png'), dpi=150)
plt.show()



# %%
# --- Comparison plot: all priors from output/ CSVs ---

output_dir = os.path.join(PROJECT_ROOT, 'output')
csv_files = sorted(Path(output_dir).glob('*_benchmark_results.csv'))

colors = plt.cm.tab10.colors

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})

ax.set_extent([-126, -113, 31, 44], crs=proj)
ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

ax.scatter(ref_final['posterior_lon'], ref_final['posterior_lat'],
           s=10, color='black', alpha=0.4, transform=proj,
           label='Reference (smooth_seismicity)', zorder=1)
ax.scatter(stations_df['longitude'], stations_df['latitude'],
           s=20, color='orange', edgecolor='k', alpha=0.7, marker='v', transform=proj,
           label='Stations', zorder=0)

for i, csv_path in enumerate(csv_files):
    prior_name = csv_path.stem.replace('_benchmark_results', '')

    df = pd.read_csv(csv_path)
    final = df.groupby('event_id').last().reset_index()
    ax.scatter(final['posterior_lon'], final['posterior_lat'],
               s=8, color=colors[i % len(colors)], alpha=0.2,
               transform=proj, label=prior_name, zorder=2)

ax.set_title('bEPIC final locations — prior comparison')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'figures', 'comparison_benchmark_locations.png'), dpi=150)
plt.show()

# %%

"""
Zoom in on Mendocino Triple Junction (MTJ) — 2x3 grid, one prior per panel
"""

lat_min = 39.0
lat_max = 42.0
lon_min = -126
lon_max = -122.5

def in_extent(df):
    return df[
        df['posterior_lat'].between(lat_min, lat_max) &
        df['posterior_lon'].between(lon_min, lon_max)
    ]

ref_mtj = in_extent(ref_final)
stations_mtj = stations_df[
    stations_df['latitude'].between(lat_min, lat_max) &
    stations_df['longitude'].between(lon_min, lon_max)
]

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform']

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={'projection': proj})

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

    ax.scatter(ref_mtj['posterior_lon'], ref_mtj['posterior_lat'],
               s=18, color='black', alpha=0.5, transform=proj,
               label='Reference', zorder=1)
    ax.scatter(stations_mtj['longitude'], stations_mtj['latitude'],
               s=40, color='orange', edgecolor='k', alpha=0.85, marker='v',
               transform=proj, label='Stations', zorder=3)

    csv_path = os.path.join(PROJECT_ROOT, 'output',
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        final = in_extent(df.groupby('event_id').last().reset_index())
        ax.scatter(final['posterior_lon'], final['posterior_lat'],
                   s=18, color='crimson', alpha=0.5, transform=proj,
                   label=prior_name, zorder=2)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.legend(loc='upper right', fontsize=7)

fig.suptitle('bEPIC MTJ locations — prior comparison', fontsize=14)
plt.tight_layout()
os.makedirs(os.path.join(PROJECT_ROOT, 'figures'), exist_ok=True)
plt.savefig(os.path.join(PROJECT_ROOT, 'figures', 'MTJ_grid_benchmark_locations.png'), dpi=150)
plt.show()

# %%
"""
MTJ misfit histograms — 2x3 grid, one prior per panel
best_misfit from final version of each event, filtered to MTJ extent
"""

ref_misfits = in_extent(ref_final)['best_misfit']

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(PROJECT_ROOT, 'output',
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        misfits = in_extent(df.groupby('event_id').last().reset_index())['best_misfit']
        ax.hist(misfits, bins=30, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.hist(ref_misfits, bins=30, color='black', alpha=0.4, label='Reference')
    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('best_misfit', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC MTJ misfit distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'figures', 'MTJ_grid_misfit_histograms.png'), dpi=150)
plt.show()

"""
TOTAL misfit histograms — 2x3 grid, one prior per panel
best_misfit from final version of each event, filtered to MTJ extent
"""

ref_misfits = ref_final['best_misfit']

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    
    csv_path = os.path.join(PROJECT_ROOT, 'output',
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        misfits = df.groupby('event_id').last().reset_index()['best_misfit']
        ax.hist(misfits, bins=30, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.hist(ref_misfits, bins=30, color='black', alpha=0.4, label='Reference')
    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('best_misfit', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC misfit distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'figures', 'Grid_misfit_histograms.png'), dpi=150)
plt.show()

# %%
