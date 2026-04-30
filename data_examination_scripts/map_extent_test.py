#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -- Spatial region (California + PNW benchmark polygon, [lat, lon] pairs) --
shape_coords = np.array([
[43.5, -127.7], [43.5, -117.5], [39.7, -117.5], [36.1, -112.6],
[34.6, -111.6], [34.3, -111.6], [32.7, -112.1], [31.8, -112.2],
[31.2, -113.5], [31.0, -117.1], [31.1, -117.4], [31.5, -118.3],
[32.4, -118.8], [33.3, -122.3], [34.0, -124.0], [37.5, -126.3],
[40.0, -127.9], [40.5, -127.9], [43.0, -127.7], [43.5, -127.7],
])

extent = (-129, -112, 30, 45)
proj   = ccrs.PlateCarree()
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})
ax.set_extent(extent, crs=proj)
ax.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor='black')
#ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
#ax.add_feature(cfeature.LAND,      facecolor='lightgray',   zorder=0)
#ax.add_feature(cfeature.OCEAN,     facecolor='lightyellow', zorder=0)
ax.plot(shape_coords[:,1],shape_coords[:,0],linewidth=2,color='r',transform=proj,zorder=2)
plt.show()

