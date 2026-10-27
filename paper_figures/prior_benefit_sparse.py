#%%
# =============================================================================
# prior_benefit_sparse.py — paper figure: a prior rescuing a poorly constrained location
# =============================================================================
# Event 203708 (M4.0, 38.05N 118.75W near Mono Lake): on land, >30 km from the coast, sparse
# seismicity (285 background events within 50 km). At 4 triggers the likelihood alone (= uniform
# prior) is 73 km off; the KDE_Seismicity prior pulls the posterior to ~1 km. Chosen by re-running
# bEPIC (uniform vs KDE) over all catalog events (the max_trigs_4 benchmark CSVs were stale).
# Columns: posterior | likelihood | prior.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from priors import SeismicPrior
from benchmark import config
from benchmark.metrics import _haversine_km as haversine_km
from benchmark.runner import (run_single_event_get_grid, load_station_availability_cache,
                              load_reference_catalog)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

N_TRIGGERS = 4
EVENTS = [(203708, 'M4.0 near Mono Lake\nprior fixes a sparse-station location')]
KDE_PATH = os.path.join(SeismicPrior.data_dir, 'kde_seismicity_benchmark.tt3')
UNI_PATH = os.path.join(SeismicPrior.data_dir, 'USGS_NSHM_prior.tt3')   # grid only; use_prior=False
RUN_DIR  = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
AVAIL    = load_station_availability_cache(os.path.join(
    PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet'))
CATALOG  = load_reference_catalog(os.path.join(
    PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')).set_index('event_id')

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'figure.dpi': 150})
proj = ccrs.PlateCarree()

#%%
# ── Run bEPIC (KDE prior and uniform) for each event at N_TRIGGERS ────────────
kde_prior = SeismicPrior.from_tt3(KDE_PATH)
uni_prior = SeismicPrior.from_tt3(UNI_PATH)
size = 2 * config.BENCHMARK_PARAMS['grid_size'] + 1

results = []
for ev, _ in EVENTS:
    run_path = os.path.join(RUN_DIR, f'{ev}.run')
    run_df = pd.read_csv(run_path)
    run_df.columns = [c.replace(' ', '_') for c in run_df.columns]
    fv = next(v for v in sorted(run_df.version.unique()) if (run_df.version == v).sum() >= N_TRIGGERS)   # as the benchmark: n_trigs = rows
    trig = run_df[run_df.version == fv].drop_duplicates('station')
    kw = dict(focus_version=fv, station_inventory=AVAIL.get(str(ev)))
    _, odf, _ = run_single_event_get_grid(run_path, kde_prior, True,  dict(config.BENCHMARK_PARAMS), **kw)
    g = lambda c: odf[c].values.reshape(size, size)
    usgs = CATALOG.loc[ev, ['usgs_lat', 'usgs_lon']].astype(float).values
    i = odf['post'].values.argmax(); j = odf['like'].values.argmax()
    err = lambda k: float(haversine_km(usgs[0], usgs[1], odf.lat.values[k], odf.lon.values[k]))
    results.append(dict(lat=g('lat'), lon=g('lon'), prior=g('prior'), like=g('like'), post=g('post'),
                        trig=trig, usgs=usgs, post_pt=(odf.lat.values[i], odf.lon.values[i]),
                        like_pt=(odf.lat.values[j], odf.lon.values[j]), post_err=err(i), like_err=err(j)))
    print(ev, f'likelihood-only {results[-1]["like_err"]:.0f} km, KDE posterior {results[-1]["post_err"]:.0f} km')

#%%
# ── Figure ───────────────────────────────────────────────────────────────────
N = len(EVENTS); FW, FH = 15, 5 * N
W, H, X0, GAP = 0.29, 0.72 / N, 0.07, 0.03
AX_POS = {(r, c): [X0 + c * (W + GAP), 0.08 + (N - 1 - r) * 0.92 / N, W, H] for r in range(N) for c in range(3)}

fig = plt.figure(figsize=(FW, FH))
for r, ((ev, rowlabel), R) in enumerate(zip(EVENTS, results)):
    # zoom to the stations, USGS epicenter and both MAP points (row 1's likelihood is a spot ~2 grid cells wide, hidden under the MAP dot)
    plats = np.r_[R['trig'].latitude, R['usgs'][0], R['post_pt'][0], R['like_pt'][0]]
    plons = np.r_[R['trig'].longitude, R['usgs'][1], R['post_pt'][1], R['like_pt'][1]]
    cy, cx = (plats.min() + plats.max()) / 2, (plons.min() + plons.max()) / 2
    asp = AX_POS[(r, 0)][2] * FW / (AX_POS[(r, 0)][3] * FH)            # axes width / height
    hl = max((plats.max() - plats.min()) / 2, (plons.max() - plons.min()) / 2 / asp) + 0.4
    ext = [cx - asp * hl, cx + asp * hl, cy - hl, cy + hl]
    panels = [(f'Posterior  ({R["post_err"]:.0f} km error)', R['post'], R['post_pt']),
              (f'Likelihood  ({R["like_err"]:.0f} km error)', R['like'], R['like_pt']),
              ('Prior  (KDE Seismicity)', R['prior'], None)]
    for c, (title, grid, pt) in enumerate(panels):
        ax = fig.add_axes(AX_POS[(r, c)], projection=proj)
        ax.set_extent(ext, crs=proj)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=1)
        ax.contour(R['lon'], R['lat'], grid / grid.max(), levels=np.linspace(0.1, 1.0, 10),
                   cmap='Reds', transform=proj, zorder=4, linewidths=1.2)
        ax.scatter(R['trig'].longitude, R['trig'].latitude, marker='^', s=50, facecolor='white',
                   edgecolor='black', linewidths=0.8, transform=proj, zorder=5,
                   label='triggered station')
        if pt is not None:
            ax.plot(pt[1], pt[0], 'o', ms=9, color='red', markeredgecolor='black', transform=proj,
                    zorder=6, label='bEPIC MAP')
        ax.plot(R['usgs'][1], R['usgs'][0], '*', ms=16, color='gold', markeredgecolor='black',
                markeredgewidth=0.8, transform=proj, zorder=7, label='USGS location')
        ax.set_title(title)
fig.savefig(os.path.join(OUTPUT_DIR, 'prior_benefit_sparse.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'prior_benefit_sparse.pdf'), bbox_inches='tight')

