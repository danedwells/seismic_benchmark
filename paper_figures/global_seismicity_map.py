#%%
# =============================================================================
# global_seismicity_map.py — global M>=4 seismicity, last 5 years (USGS ComCat)
# =============================================================================
# Same base map as eew_world_map.py, with the EEW regions shaded behind the seismicity, to contrast
# where EEW systems exist with where earthquakes occur. Catalog is cached after the first download.
# =============================================================================

import os
import io
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

OUT   = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(OUT), 'data', 'global_m4_5yr.csv')
MIN_MAG = 4.0
EEW_COLOR = '#c0392b'
EEW_COUNTRIES = ['Japan', 'China', 'Taiwan', 'South Korea', 'Israel']
EEW_REGIONS = {   # admin-0 name -> admin-1 names (same as eew_world_map.py)
    'United States of America': ['California', 'Oregon', 'Washington'],
    'Canada': ['British Columbia'],
    'Mexico': ['Distrito Federal', 'Ciudad de México', 'México', 'Morelos', 'Puebla', 'Guerrero',
               'Oaxaca', 'Michoacán', 'Colima', 'Jalisco', 'Chiapas'],
}
END   = pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=5)

def download():
    edges = pd.date_range(START, END, freq='MS').union([START, END])
    dfs = []
    for a, b in zip(edges[:-1], edges[1:]):
        url = ('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc'
               f'&minmagnitude={MIN_MAG}&starttime={a:%Y-%m-%d}&endtime={b:%Y-%m-%d}')
        d = pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
        assert len(d) < 20000, f'{a:%Y-%m}: hit the 20,000-event API limit'
        dfs.append(d[['time', 'latitude', 'longitude', 'mag']])
    pd.concat(dfs).drop_duplicates().to_csv(CACHE, index=False)

if not os.path.exists(CACHE):
    download()
cat = pd.read_csv(CACHE)
print(f'{len(cat)} events M>={MIN_MAG}, {cat.time.min()[:10]} to {cat.time.max()[:10]}')

fig = plt.figure(figsize=(13, 6.5))
ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=0))
ax.set_global()
ax.set_ylim(-6.2e6, 8.4e6)
ax.add_feature(cfeature.LAND, facecolor='#ececec', edgecolor='none')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='white', linewidth=0.4)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#999999', linewidth=0.3)
ax.spines['geo'].set_visible(False)

countries = list(shpreader.Reader(shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')).records())
states    = list(shpreader.Reader(shpreader.natural_earth('10m', 'cultural', 'admin_1_states_provinces')).records())
eew = [c.geometry for c in countries if c.attributes['NAME'] in EEW_COUNTRIES or c.attributes['ADMIN'] in EEW_COUNTRIES]
for adm, names in EEW_REGIONS.items():
    eew += [st.geometry for st in states if st.attributes['admin'] == adm and st.attributes['name'] in names]
ax.add_geometries(eew, ccrs.PlateCarree(), facecolor=EEW_COLOR, edgecolor='white', linewidth=0.4, zorder=1)

cat = cat.sort_values('mag')   # large events on top
ax.scatter(cat.longitude, cat.latitude, s=1.5 * (cat.mag - 3) ** 2, c='#1f3a5f', alpha=0.35,
           linewidths=0, transform=ccrs.PlateCarree(), zorder=2)

fig.savefig(os.path.join(OUT, 'figures', 'global_seismicity_map.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figures', 'global_seismicity_map.pdf'), bbox_inches='tight')
