#%% #!/usr/bin/env python3
"""
catalog_examination.py

Standalone examination of the bEPIC testing catalog.

Sections
--------
1. Load catalog and parse dates
2. Summary statistics
3. Map of USGS/ANSS locations (colored by magnitude)
4. Magnitude vs. time
5. Temporal evolution map (events colored by time)
6. USGS online verification — compares catalog lat/lon/depth/mag/time
   against live USGS ComCat queries for a sample of events.

Configuration knobs are at the top of each section.
"""

#%%
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timezone

from benchmark.runner import load_reference_catalog, get_usgs_event

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'reference', 'bEPIC_testing_catalog.txt')


#%%
# ---------------------------------------------------------------------------
# 1. Load catalog and parse dates
# ---------------------------------------------------------------------------
raw = pd.read_csv(CATALOG_PATH, sep='\t')

# Parse 'ANSS date' — format: '2018-09-30-14:41:29.510-GMT'
raw['datetime'] = pd.to_datetime(
    raw['ANSS date'].str.replace(r'-GMT$', '', regex=True),
    format='%Y-%m-%d-%H:%M:%S.%f',
    utc=True,
)

catalog = load_reference_catalog(CATALOG_PATH)
catalog['datetime'] = pd.to_datetime(raw['datetime'], utc=True)

catalog = catalog.sort_values('datetime').reset_index(drop=True)

print(f"Catalog loaded: {len(catalog)} events")
print(f"  Date range : {catalog['datetime'].iloc[0].date()}  →  {catalog['datetime'].iloc[-1].date()}")
print(f"  Mag range  : {catalog['usgs_mag'].min():.1f} – {catalog['usgs_mag'].max():.1f}")
print(f"  Depth range: {catalog['usgs_depth'].min():.1f} – {catalog['usgs_depth'].max():.1f} km")
print(f"  Lat range  : {catalog['usgs_lat'].min():.3f} – {catalog['usgs_lat'].max():.3f}")
print(f"  Lon range  : {catalog['usgs_lon'].min():.3f} – {catalog['usgs_lon'].max():.3f}")


#%%
# ---------------------------------------------------------------------------
# 2. Map of USGS/ANSS locations — colored by magnitude
# ---------------------------------------------------------------------------
proj = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': proj})
ax.set_extent([catalog['usgs_lon'].min() - 1, catalog['usgs_lon'].max() + 1,
               catalog['usgs_lat'].min() - 1, catalog['usgs_lat'].max() + 1],
              crs=proj)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', alpha=0.4)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
gl.top_labels   = False
gl.right_labels = False

sc = ax.scatter(
    catalog['usgs_lon'], catalog['usgs_lat'],
    c=catalog['usgs_mag'], cmap='plasma',
    s=2 * (catalog['usgs_mag'] - catalog['usgs_mag'].min() + 0.1) ** 3,
    alpha=0.4, transform=proj, zorder=5,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('ANSS Magnitude', fontsize=10)
ax.set_title(f'USGS/ANSS Event Locations  (n={len(catalog)})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'data', 'reference', 'catalog_map.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# ---------------------------------------------------------------------------
# 3. Magnitude distribution as a function of time
# ---------------------------------------------------------------------------
# Manual axes positions: [left, bottom, width, height] in figure fractions.
# Adjust these to reposition any panel or the colorbar.
AX_POS = {
    'mag':    [0.07, 0.52, 0.82, 0.44],   # lollipop: magnitude vs time
    'cbar':   [0.91, 0.52, 0.015, 0.44],  # colorbar for depth
    'count':  [0.07, 0.30, 0.82, 0.21],   # cumulative event count
    'moment': [0.07, 0.07, 0.82, 0.21],   # cumulative seismic moment
}

fig = plt.figure(figsize=(13, 9))

# Seismic moment: M₀ = 10^(1.5 × Mw + 9.1)  [N·m]
catalog['moment'] = 10 ** (1.5 * catalog['usgs_mag'] + 9.1)
cumulative_moment = catalog['moment'].cumsum()

# --- panel 1: lollipop magnitude vs time, colored by depth ---
ax1 = fig.add_axes(AX_POS['mag'])
depth_norm = mcolors.Normalize(vmin=catalog['usgs_depth'].min(),
                                vmax=catalog['usgs_depth'].max())
depth_cmap = plt.cm.viridis_r
colors = depth_cmap(depth_norm(catalog['usgs_depth'].values))

# Lollipop: vertical stems from y=0 to magnitude, dot at top
ax1.vlines(catalog['datetime'], 0, catalog['usgs_mag'],
           color=colors, linewidth=0.8, alpha=0.7)
sc = ax1.scatter(catalog['datetime'], catalog['usgs_mag'],
                 c=catalog['usgs_depth'], cmap=depth_cmap, norm=depth_norm,
                 s=12, zorder=3)
ax1.set_ylabel('ANSS Magnitude', fontsize=10)
ax1.set_title('Magnitude vs. Time', fontsize=11)
ax1.set_ylim(bottom=0)
ax1.grid(True, linewidth=0.3, alpha=0.4, axis='y')
ax1.tick_params(labelbottom=False)

cax = fig.add_axes(AX_POS['cbar'])
fig.colorbar(sc, cax=cax).set_label('Depth (km)', fontsize=9)

# Monthly event count bars on twin axis
ax1b = ax1.twinx()
mag_monthly = catalog.set_index('datetime')['usgs_mag'].resample('ME').count()
ax1b.bar(mag_monthly.index, mag_monthly.values, width=20, alpha=0.12,
         color='steelblue')
ax1b.set_ylabel('Events / month', fontsize=9, color='steelblue')
ax1b.tick_params(axis='y', labelcolor='steelblue', labelsize=8)

# --- panel 2: cumulative event count ---
ax2 = fig.add_axes(AX_POS['count'], sharex=ax1)
ax2.plot(catalog['datetime'], np.arange(1, len(catalog) + 1),
         color='black', linewidth=1.2)
ax2.set_ylabel('Cumulative\nevents', fontsize=9)
ax2.grid(True, linewidth=0.3, alpha=0.4)
ax2.tick_params(labelbottom=False)

# --- panel 3: cumulative seismic moment ---
ax3 = fig.add_axes(AX_POS['moment'], sharex=ax1)
ax3.plot(catalog['datetime'], cumulative_moment, color='darkred', linewidth=1.2)
ax3.set_ylabel('Cumulative M₀ (N·m)', fontsize=9)
ax3.set_xlabel('Date', fontsize=10)
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x:.1e}')
)
ax3.grid(True, linewidth=0.3, alpha=0.4)

plt.savefig(os.path.join(PROJECT_ROOT, 'data', 'reference', 'catalog_magnitude_time.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# ---------------------------------------------------------------------------
# 4. Temporal evolution map — single panel, events colored by origin time
# ---------------------------------------------------------------------------
import matplotlib.dates as mdates

# Convert to matplotlib date numbers (days since 0001-01-01) — float64 has
# sufficient precision at this scale, unlike nanosecond int64 values.
date_nums = mdates.date2num(catalog['datetime'].dt.to_pydatetime())
time_norm = mcolors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
cmap = plt.cm.plasma

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})
ax.set_extent([catalog['usgs_lon'].min() - 0.5, catalog['usgs_lon'].max() + 0.5,
               catalog['usgs_lat'].min() - 0.5, catalog['usgs_lat'].max() + 0.5],
              crs=proj)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', alpha=0.4)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke')

sc = ax.scatter(
    catalog['usgs_lon'], catalog['usgs_lat'],
    c=date_nums, norm=time_norm, cmap=cmap,
    s=18, alpha=0.7, transform=proj, zorder=5,
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
cbar.set_label('Origin time', fontsize=10)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
gl.top_labels   = False
gl.right_labels = False

ax.set_title(f'Temporal Evolution of Catalog Events  (n={len(catalog)})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'data', 'reference', 'catalog_temporal_evolution.png'), dpi=150, bbox_inches='tight')
plt.show()


#%%
# ---------------------------------------------------------------------------
# 5. USGS online verification
# ---------------------------------------------------------------------------
# Set N_VERIFY to an integer to check a random sample, or None to verify all.
# Queries are rate-limited to ~2/second to be polite to the USGS API.
N_VERIFY     = 30        # number of events to verify (None = all)
REQUEST_DELAY = 0.5      # seconds between requests
MAG_TOL      = 0.3       # flag if |catalog_mag  - usgs_mag|  > this
LOC_TOL_DEG  = 0.05      # flag if location offset > this many degrees
DEPTH_TOL_KM = 5.0       # flag if |catalog_depth - usgs_depth| > this
TIME_TOL_S   = 2.0       # flag if |catalog_time  - usgs_time|  > this (seconds)

sample = catalog if N_VERIFY is None else catalog.sample(n=N_VERIFY, random_state=42)
sample = sample.sort_values('datetime').reset_index(drop=True)

print(f"\nVerifying {len(sample)} events against USGS ComCat …")

rows = []
for _, row in sample.iterrows():
    result = {
        'event_id':    row['event_id'],
        'anss_id':     row['anss_id'],
        'cat_mag':     row['usgs_mag'],
        'cat_lat':     row['usgs_lat'],
        'cat_lon':     row['usgs_lon'],
        'cat_depth':   row['usgs_depth'],
        'cat_time':    row['datetime'],
        'usgs_mag':    None, 'usgs_lat': None, 'usgs_lon': None,
        'usgs_depth':  None, 'usgs_time': None,
        'd_mag': None, 'd_lat': None, 'd_lon': None,
        'd_depth': None, 'd_time_s': None,
        'status': 'ok',
    }
    try:
        gj    = get_usgs_event(row['anss_id'])
        props = gj['properties']
        coords = gj['geometry']['coordinates']  # [lon, lat, depth_km]

        usgs_time = pd.Timestamp(props['time'], unit='ms', tz='UTC')

        result.update({
            'usgs_mag':   props['mag'],
            'usgs_lat':   coords[1],
            'usgs_lon':   coords[0],
            'usgs_depth': coords[2],
            'usgs_time':  usgs_time,
            'd_mag':      props['mag']  - row['usgs_mag'],
            'd_lat':      coords[1]     - row['usgs_lat'],
            'd_lon':      coords[0]     - row['usgs_lon'],
            'd_depth':    coords[2]     - row['usgs_depth'],
            'd_time_s':   (usgs_time - row['datetime']).total_seconds(),
        })

        flags = []
        if abs(result['d_mag'])    > MAG_TOL:      flags.append('mag')
        if abs(result['d_lat'])    > LOC_TOL_DEG:  flags.append('lat')
        if abs(result['d_lon'])    > LOC_TOL_DEG:  flags.append('lon')
        if abs(result['d_depth'])  > DEPTH_TOL_KM: flags.append('depth')
        if abs(result['d_time_s']) > TIME_TOL_S:   flags.append('time')
        result['status'] = ', '.join(flags) if flags else 'ok'

    except Exception as e:
        result['status'] = f'ERROR: {e}'

    rows.append(result)
    time.sleep(REQUEST_DELAY)

verify_df = pd.DataFrame(rows)

# --- summary ---
n_ok      = (verify_df['status'] == 'ok').sum()
n_flagged = (verify_df['status'] != 'ok').sum()
n_errors  = verify_df['status'].str.startswith('ERROR').sum()

print(f"\nVerification complete: {n_ok} ok  |  {n_flagged} flagged  |  {n_errors} errors")

ok_mask = verify_df['d_mag'].notna()
if ok_mask.any():
    v = verify_df[ok_mask]
    print(f"\n  |Δmag|    mean={v['d_mag'].abs().mean():.3f}  max={v['d_mag'].abs().max():.3f}")
    print(f"  |Δlat|    mean={v['d_lat'].abs().mean():.4f}°  max={v['d_lat'].abs().max():.4f}°")
    print(f"  |Δlon|    mean={v['d_lon'].abs().mean():.4f}°  max={v['d_lon'].abs().max():.4f}°")
    print(f"  |Δdepth|  mean={v['d_depth'].abs().mean():.2f} km  max={v['d_depth'].abs().max():.2f} km")
    print(f"  |Δtime|   mean={v['d_time_s'].abs().mean():.2f} s   max={v['d_time_s'].abs().max():.2f} s")

flagged = verify_df[verify_df['status'] != 'ok']
if not flagged.empty:
    print(f"\nFlagged events:")
    print(flagged[['anss_id', 'cat_mag', 'd_mag', 'd_lat', 'd_lon',
                    'd_depth', 'd_time_s', 'status']].to_string(index=False))

# Save verification table
out_path = os.path.join(PROJECT_ROOT, 'data', 'reference', 'catalog_verification.csv')
verify_df.to_csv(out_path, index=False)
print(f"\nVerification results saved to {out_path}")

# %%
