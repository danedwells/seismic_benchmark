#%%
# =============================================================================
# eew_world_map.py — where public earthquake early warning systems operate
# =============================================================================
# Whole countries are shaded where the system is national; where coverage is
# regional (US West Coast, British Columbia, Mexico's SASMEX states) only the
# relevant states/provinces are shaded. Android Earthquake Alerts excluded.
# =============================================================================

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

OUT = os.path.dirname(os.path.abspath(__file__))
COLOR = '#c0392b'

COUNTRIES = ['Japan', 'China', 'Taiwan', 'South Korea', 'Israel']
REGIONS = {   # admin-0 name -> admin-1 names
    'United States of America': ['California', 'Oregon', 'Washington'],
    'Canada': ['British Columbia'],
    'Mexico': ['Distrito Federal', 'Ciudad de México', 'México', 'Morelos', 'Puebla', 'Guerrero',
               'Oaxaca', 'Michoacán', 'Colima', 'Jalisco', 'Chiapas'],
}

countries = list(shpreader.Reader(shpreader.natural_earth('50m', 'cultural', 'admin_0_countries')).records())
states    = list(shpreader.Reader(shpreader.natural_earth('10m', 'cultural', 'admin_1_states_provinces')).records())

fig = plt.figure(figsize=(13, 6.5))
ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=0))
ax.set_global()
ax.set_ylim(-6.2e6, 8.4e6)
ax.add_feature(cfeature.LAND, facecolor='#ececec', edgecolor='none')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='white', linewidth=0.4)
ax.spines['geo'].set_visible(False)

def fill(geoms):
    ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor=COLOR, edgecolor='white', linewidth=0.4)

fill([c.geometry for c in countries if c.attributes['NAME'] in COUNTRIES or c.attributes['ADMIN'] in COUNTRIES])
for adm, names in REGIONS.items():
    fill([s.geometry for s in states if s.attributes['admin'] == adm and s.attributes['name'] in names])

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#999999', linewidth=0.3)
fig.savefig(os.path.join(OUT, 'figures', 'eew_world_map.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figures', 'eew_world_map.pdf'), bbox_inches='tight')
