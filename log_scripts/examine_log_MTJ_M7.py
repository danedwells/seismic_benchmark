"""
Diagnose the location error of the ETAS (sigma_and_prior) EPIC replay of the 2024 M7.0 MTJ
mainshock (nc75095651).  Copy of examine_log.py, tailored to this event.  Uses every EPIC
version with eventid > 0 (alerted or not); the sent alert messages are still marked on the map.

Compares each alerted version's location/magnitude to the USGS/ANSS solution and plots the
location evolution on a map plus error (total and east/north components), station count and
magnitude vs. time since origin.
"""
#%%
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xml.etree.ElementTree as ET
from obspy.geodetics import gps2dist_azimuth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.usgs import get_usgs_event

RUN = 'MTJ_sigma_and_prior'      # ETAS; 'MTJ_control' is the KDE_Seismicity run
LOG = Path(__file__).resolve().parents[1] / 'log_files' / 'MTJ-selected' / RUN / 'epic' / 'nc75095651.log'
OUT = Path(__file__).resolve().parent / 'figures' / f'{RUN}_{LOG.stem}_examine.png'

#%% Parse
EVENT_COLS = ('eventid ver evlat evlon dep mag time latu lonu depu magu timeu lk nTb nSb nT nS '
              'ave rms fitok splitok near statrig active inact nsta percnt prcntok mindist maxdist '
              'distok azspan Mok nSok Lok Tdif tpave pdave TF Tok Azok Aok Ast alert_time').split()

events, alerts, cur_xml = [], [], None
for line in LOG.read_text().splitlines():
    msg = line.split('| INFO | ', 1)[-1]
    # 'E:I:F:' is the event's FIRST alert version; same columns as 'E:I:', just a different tag
    if msg.startswith('E:I:F:'):
        events.append(msg[6:].split())
    elif msg.startswith('E:I:  ') and not msg.startswith('E:I:T') and 'eventid' not in msg:
        events.append(msg[4:].split())
    elif msg.startswith('<event_message'):
        cur_xml = [msg]
    elif cur_xml is not None:
        cur_xml.append(msg)
        if msg.startswith('</event_message'):
            alerts.append(ET.fromstring('\n'.join(cur_xml)))
            cur_xml = None

ev = pd.DataFrame(events, columns=EVENT_COLS)
for c in ev.columns:
    if c not in ('time', 'alert_time'):
        ev[c] = pd.to_numeric(ev[c])
ev['time'] = pd.to_datetime(ev['time'])
ev['alert_time'] = pd.to_datetime(ev['alert_time'])

al = pd.DataFrame([{
    'type': a.get('message_type'), 'version': int(a.get('version')),
    'mag': float(a.findtext('core_info/mag')), 'lat': float(a.findtext('core_info/lat')),
    'lon': float(a.findtext('core_info/lon')), 'depth': float(a.findtext('core_info/depth')),
    'orig_time': pd.to_datetime(a.findtext('core_info/orig_time')).tz_localize(None),
    'num_stations': int(a.findtext('core_info/num_stations')),
    'likelihood': float(a.findtext('core_info/likelihood')),
} for a in alerts])

#%% USGS reference
anss_id = LOG.stem
props = get_usgs_event(anss_id)
usgs_lon, usgs_lat, usgs_dep = props['geometry']['coordinates']
usgs_mag = props['properties']['mag']
usgs_time = pd.to_datetime(props['properties']['time'], unit='ms')

ev = ev[ev.eventid > 0].copy()          # eventid < 0 = never alerted
ev['err_km'] = [gps2dist_azimuth(usgs_lat, usgs_lon, la, lo)[0] / 1e3
                for la, lo in zip(ev.evlat, ev.evlon)]
ev['dmag'] = ev.mag - usgs_mag
# Signed offset of the EPIC location from USGS (km): where the error points
ev['east_km']  = (ev.evlon - usgs_lon) * 111.19 * np.cos(np.radians(usgs_lat))
ev['north_km'] = (ev.evlat - usgs_lat) * 111.19
# NOTE: alert_time is data time (when the alert criteria were checked on the
# replayed-packet timeline), NOT processing order. Rows are in version order, but
# trigger latencies differ across stations/networks, so late-arriving triggers with
# earlier timestamps produce versions whose alert_time is *earlier* than the previous
# version's (11 of 96 steps in nc75098996). Lines plotted against t_since_origin
# therefore double back in time. This is unavoidable latency in the data; we plot it
# as-is rather than sorting or smoothing it away.
ev['t_since_origin'] = (ev.alert_time - usgs_time).dt.total_seconds()
al['err_km'] = [gps2dist_azimuth(usgs_lat, usgs_lon, la, lo)[0] / 1e3 for la, lo in zip(al.lat, al.lon)]
al['dmag'] = al.mag - usgs_mag
al['t_since_origin'] = (al.orig_time - usgs_time).dt.total_seconds()

#%% Summary
first_ok = ev[ev.Aok == 1].iloc[0]
print(f'USGS {anss_id}: M{usgs_mag} ({usgs_lat:.4f}, {usgs_lon:.4f}, {usgs_dep} km) {usgs_time}')
print(f'{len(ev)} versions with eventid>0, {len(al)} alert messages sent')
print(f'First alertable version: ver={first_ok.ver:.0f}, {first_ok.t_since_origin:.1f} s after origin, '
      f'nS={first_ok.nS:.0f}, M={first_ok.mag}, err={first_ok.err_km:.1f} km')
print(f'Final version: M={ev.mag.iloc[-1]} (dM={ev.dmag.iloc[-1]:+.2f}), err={ev.err_km.iloc[-1]:.1f} km, nS={ev.nS.iloc[-1]:.0f}')
print(ev[['ver', 'nT', 'nS', 'mag', 'dmag', 'evlat', 'evlon', 'err_km', 'east_km', 'north_km',
          't_since_origin']].round(2).to_string(index=False))
print(al[['type', 'version', 'mag', 'lat', 'lon', 'num_stations', 'likelihood', 'err_km', 'dmag']].to_string())

#%% Plot
def add_base(ax):
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)
    ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

PC = ccrs.PlateCarree()
fig = plt.figure(figsize=(17, 5.5))
ax_map = fig.add_subplot(1, 3, 1, projection=PC)
ax_err = fig.add_subplot(1, 3, 2)
ax_mag = fig.add_subplot(1, 3, 3)

# Panel 1: map, ~1 deg half-width around USGS location
pad = 1.0
ax_map.set_extent([usgs_lon - pad, usgs_lon + pad, usgs_lat - pad, usgs_lat + pad], crs=PC)
add_base(ax_map)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6)
gl.top_labels = gl.right_labels = False
sc = ax_map.scatter(ev.evlon, ev.evlat, c=ev.t_since_origin, cmap='viridis', s=20, transform=PC, zorder=3, label='EPIC versions')
ax_map.scatter(al.lon, al.lat, facecolors='none', edgecolors='k', s=90, transform=PC, zorder=4, label='Alerts sent')
ax_map.plot(usgs_lon, usgs_lat, 'r*', ms=18, transform=PC, zorder=5, label=f'USGS M{usgs_mag}')
fig.colorbar(sc, ax=ax_map, label='alert time since origin (s)', shrink=0.8)
ax_map.legend(loc='lower left')
ax_map.set_title(f'{anss_id} location evolution')

# Inset: California context
inset = ax_map.inset_axes([0.62, 0.62, 0.36, 0.36], projection=PC)
inset.set_extent([-125.5, -113.5, 32, 42.5], crs=PC)
add_base(inset)
inset.plot(usgs_lon, usgs_lat, 'r*', ms=10, transform=PC)

# Panel 2: location error and station count
ax_err.plot(ev.t_since_origin, ev.err_km, 'r.-', label='location error')
ax_err.plot(ev.t_since_origin, ev.east_km, 'r--', lw=1, label='east of USGS')
ax_err.plot(ev.t_since_origin, ev.north_km, 'r:', lw=1.5, label='north of USGS')
ax_err.set_xlabel('Time since origin (s)'); ax_err.set_ylabel('Distance to USGS (km)', color='r')
ax_ns = ax_err.twinx()
ax_ns.plot(ev.t_since_origin, ev.nS, 'g.-', label='nS (stations)')
ax_ns.set_ylabel('Stations in alert')
h1, l1 = ax_err.get_legend_handles_labels(); h2, l2 = ax_ns.get_legend_handles_labels()
ax_err.legend(h1 + h2, l1 + l2, loc='center right')
ax_err.set_title('Location error')

# Panel 3: magnitude
ax_mag.plot(ev.t_since_origin, ev.mag, 'b.-', label='EPIC Mw')
ax_mag.axhline(usgs_mag, color='k', ls='--', label=f'USGS M{usgs_mag}')
ax_mag.set_xlabel('Time since origin (s)'); ax_mag.set_ylabel('Magnitude')
ax_mag.legend(); ax_mag.set_title('Magnitude')

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print('Saved', OUT)

# %%
