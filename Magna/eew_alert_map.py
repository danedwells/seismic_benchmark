#%%
# =============================================================================
# eew_alert_map.py — hypothetical EEW alert times for the Magna, UT sequence
# =============================================================================
# Estimates when an earthquake early warning alert could have been issued for
# each Magna event, and maps what that means along the Wasatch Front.
#
#   alert time  = (arrival of the N_ALERT_TRIGS-th station's P pick - origin time)
#                 + ALERT_LATENCY_S      (telemetry + trigger + processing)
#   warning     = (hypocentral distance / Vs) - alert time      at a given site
#                 (positive = seconds of warning before S-waves arrive;
#                  negative = site is inside the "blind zone")
#
# Picks come from the .run files built by download_triggers.py (UU stations);
# epicenters/origin times are the USGS catalog values.
#
# Figure A: mainshock — warning-time map with the blind-zone contour and the
#           warning time at Wasatch Front cities.
# Figure B: whole sequence — epicenters colored by alert time.
#
# Usage: python Magna/eew_alert_map.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

MAGNA_DIR    = os.path.dirname(os.path.abspath(__file__))
RUN_DIR      = os.path.join(MAGNA_DIR, 'data', 'run_files')
CATALOG_PATH = os.path.join(MAGNA_DIR, 'data', 'Magna_2020_catalog.parquet')
FIG_DIR      = os.path.join(MAGNA_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ALERT_TRIGS   = 4      # alert when this many stations have reported a P pick
ALERT_LATENCY_S = 4.0    # seconds from the last needed P pick to the alert reaching users
VS_KMS          = 3.5    # S-wave speed (km/s)

# Map extent [lon0, lon1, lat0, lat1] along the Wasatch Front
EXTENT = [-112.8, -111.2, 39.9, 41.5]

# name: (lat, lon, label offset in points (dx, dy), horizontal alignment)
CITIES = {
    'Ogden':          (41.2230, -111.9738, ( 8,  4), 'left'),
    'Salt Lake City': (40.7608, -111.8910, ( 8,  4), 'left'),
    'West Valley':    (40.6916, -112.0011, ( 8, -18), 'left'),
    'Magna':          (40.7091, -112.1013, (-10, -16), 'right'),
    'Tooele':         (40.5308, -112.2983, ( 8,  4), 'left'),
    'Park City':      (40.6461, -111.4980, (-8, 12), 'right'),
    'Provo':          (40.2338, -111.6585, ( 8,  4), 'left'),
}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                     'axes.titlesize': 13, 'axes.titleweight': 'bold', 'figure.dpi': 150})

#%%
# ---------------------------------------------------------------------------
# Alert time per event
# ---------------------------------------------------------------------------
catalog = pd.read_parquet(CATALOG_PATH)
catalog['time'] = pd.to_datetime(catalog['time'], utc=True)
catalog = catalog.sort_values('time').reset_index(drop=True)

rows = []
for ev in catalog.itertuples(index=False):
    run_path = os.path.join(RUN_DIR, f'{ev.id}.run')
    if not os.path.exists(run_path):
        continue
    run = pd.read_csv(run_path)
    version = run[run['version'] == N_ALERT_TRIGS - 1]   # version k holds k+1 stations
    if version.empty:
        continue                                          # fewer than N_ALERT_TRIGS stations
    t_pick = version['trigger time'].max()                # the Nth station's P pick (unix s)
    t_alert = t_pick - ev.time.timestamp() + ALERT_LATENCY_S
    depth = 8.0 if pd.isna(ev.depth) else float(ev.depth)
    rows.append(dict(event_id=ev.id, mag=ev.mag, origin=ev.time, lat=ev.latitude, lon=ev.longitude,
                     depth_km=depth, alert_time_s=t_alert,
                     blind_radius_km=np.sqrt(max((t_alert * VS_KMS) ** 2 - depth ** 2, 0.0))))
alerts = pd.DataFrame(rows)
alerts.to_csv(os.path.join(FIG_DIR, 'alert_times.csv'), index=False)

main = alerts.loc[alerts['mag'].idxmax()]
print(f"{len(alerts)}/{len(catalog)} events have >= {N_ALERT_TRIGS} stations")
print(f"Sequence alert time: median {alerts['alert_time_s'].median():.1f} s, "
      f"min {alerts['alert_time_s'].min():.1f} s, max {alerts['alert_time_s'].max():.1f} s")
print(f"Mainshock {main.event_id} M{main.mag:.1f}: alert at {main.alert_time_s:.1f} s after origin; "
      f"blind-zone radius {main.blind_radius_km:.1f} km")


def hypo_dist_km(lat, lon, ev):
    """Hypocentral distance (km) from an event to point(s) (lat, lon); flat-earth approximation."""
    dx = (np.asarray(lon) - ev.lon) * 111.32 * np.cos(np.radians(ev.lat))
    dy = (np.asarray(lat) - ev.lat) * 110.57
    return np.sqrt(dx ** 2 + dy ** 2 + ev.depth_km ** 2)


def warning_s(lat, lon, ev):
    """Seconds between the alert and S-wave arrival at (lat, lon) for event ev (negative = blind zone)."""
    return hypo_dist_km(lat, lon, ev) / VS_KMS - ev.alert_time_s


print("\nWarning time at cities for the mainshock (s; negative = no warning):")
for name, (clat, clon, *_) in CITIES.items():
    print(f"  {name:15s} {warning_s(clat, clon, main):6.1f}")


def base_map(fig, rect):
    proj = ccrs.PlateCarree()
    ax = fig.add_axes(rect, projection=proj)
    ax.set_extent(EXTENT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'),
                   facecolor='#d6eaf8', edgecolor='#7fb3d5', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=3)
    ax.set_xticks(np.linspace(EXTENT[0], EXTENT[1], 5), crs=proj)
    ax.set_yticks(np.linspace(EXTENT[2], EXTENT[3], 5), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
    return ax, proj


#%%
# ---------------------------------------------------------------------------
# Figure A — mainshock warning-time map
# ---------------------------------------------------------------------------
lon_g = np.linspace(EXTENT[0], EXTENT[1], 400)
lat_g = np.linspace(EXTENT[2], EXTENT[3], 400)
LON, LAT = np.meshgrid(lon_g, lat_g)
W = warning_s(LAT, LON, main)

fig = plt.figure(figsize=(8, 9))
ax, proj = base_map(fig, [0.10, 0.07, 0.72, 0.85])

levels = np.arange(-10, 31, 2)
norm = mcolors.TwoSlopeNorm(vmin=levels[0], vcenter=0, vmax=levels[-1])
cf = ax.contourf(LON, LAT, W, levels=levels, cmap='RdYlGn', norm=norm, extend='both',
                 transform=proj, alpha=0.75, zorder=1)
cs0 = ax.contour(LON, LAT, W, levels=[0], colors='k', linewidths=1.8, transform=proj, zorder=2)
ax.clabel(cs0, fmt=lambda v: 'blind zone', fontsize=10)
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.04, pad=0.03)
cbar.set_label('Warning time before S-wave arrival (s)')

ax.scatter(main.lon, main.lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1,
           transform=proj, zorder=6, label=f'Magna M{main.mag:.1f} epicenter')
for name, (clat, clon, (dx, dy), ha) in CITIES.items():
    w = warning_s(clat, clon, main)
    ax.scatter(clon, clat, s=40, c='k', transform=proj, zorder=5)
    ax.annotate(f'{name}\n{w:+.0f} s', (clon, clat), xycoords=proj._as_mpl_transform(ax),
                xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=10,
                fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_title(f'Hypothetical EEW warning time — {main.origin:%Y-%m-%d} Magna M{main.mag:.1f}\n'
             f'alert {main.alert_time_s:.1f} s after origin '
             f'({N_ALERT_TRIGS} stations + {ALERT_LATENCY_S:g} s latency)', fontsize=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
fig.savefig(os.path.join(FIG_DIR, 'eew_warning_time_mainshock.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'eew_warning_time_mainshock.pdf'), bbox_inches='tight')
plt.show()

#%%
# ---------------------------------------------------------------------------
# Figure B — whole sequence, epicenters colored by alert time
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 9))
ax, proj = base_map(fig, [0.10, 0.07, 0.72, 0.85])

aft = alerts[alerts['event_id'] != main.event_id]
sc = ax.scatter(aft['lon'], aft['lat'], c=aft['alert_time_s'], cmap='viridis_r',
                s=15 * (aft['mag'] - 2) ** 2, edgecolors='k', linewidths=0.4,
                transform=proj, zorder=5)
ax.scatter(main.lon, main.lat, marker='*', s=450, c='yellow', edgecolors='k', linewidths=1,
           transform=proj, zorder=6, label=f'Mainshock M{main.mag:.1f} ({main.alert_time_s:.1f} s)')
ax.tissot(rad_km=main.blind_radius_km, lons=main.lon, lats=main.lat, n_samples=100,
          facecolor='none', edgecolor='red', linestyle='--', linewidth=1.8, zorder=4)
for name, (clat, clon, (dx, dy), ha) in CITIES.items():
    ax.scatter(clon, clat, s=30, c='k', marker='s', transform=proj, zorder=5)
    ax.annotate(name, (clon, clat), xycoords=proj._as_mpl_transform(ax), xytext=(dx, dy),
                textcoords='offset points', ha=ha, fontsize=10, fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.04, pad=0.03)
cbar.set_label('Alert time after origin (s)')
ax.set_title(f'Hypothetical EEW alert times — Magna sequence (n={len(alerts)}, M≥3)', fontsize=12)
handles, _ = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color='red', linestyle='--', linewidth=1.8,
                      label=f'Mainshock blind zone ({main.blind_radius_km:.0f} km radius)'))
ax.legend(handles=handles, loc='lower left', fontsize=10, framealpha=0.9)
fig.savefig(os.path.join(FIG_DIR, 'eew_alert_times_sequence.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'eew_alert_times_sequence.pdf'), bbox_inches='tight')
plt.show()
