"""
plot_cascadia_catalog.py

Standalone example: plot every earthquake in data/cascadia/reference/*.csv
on a Cartopy map, colored by depth (with a colorbar) and scaled by magnitude.

Usage
-----
    python examples/plot_cascadia_catalog.py

Requires: matplotlib, pandas, numpy, cartopy
"""
#%%
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_DIR = "/home/a01738353/2024_NEHRP/seismic_benchmark/data/cascadia/reference"
OUT_PNG  = f"{DATA_DIR}/cascadia_depth_magnitude_map.png"

# --- Load every CSV in the folder into one DataFrame ---
csv_paths = glob.glob(f"{DATA_DIR}/*.csv")
df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
df = df.dropna(subset=["latitude", "longitude", "depth", "mag"])
df['time'] = pd.to_datetime(df['time'],format="ISO8601")


# --- Map 1 ---
# Magnitude and depth
pad = 1.0
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([df["longitude"].min() - pad, df["longitude"].max() + pad,
               df["latitude"].min() - pad, df["latitude"].max() + pad],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAND, facecolor="whitesmoke")

sizes = 2 * 2 ** df["mag"]
vmin = 0 
vmax = 80
sc = ax.scatter(df["longitude"], df["latitude"],
                 s=sizes, c=df["depth"], cmap="viridis",
                 alpha=0.4, edgecolor="k", linewidth=0.2,
                 transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
cbar.set_label("Depth (km)")

ax.set_title(f"Cascadia catalog: {len(df):,} events "
             "(color = depth, size = magnitude)")
ax.gridlines(draw_labels=True, linewidth=0.3)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Wrote map to {OUT_PNG}")
plt.show()


#%%
# Map 2
# Magnitude and time
# --- Map ---
OUT_PNG  = f"{DATA_DIR}/cascadia_time_magnitude_map.png"
pad = 1.0
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([df["longitude"].min() - pad, df["longitude"].max() + pad,
               df["latitude"].min() - pad, df["latitude"].max() + pad],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAND, facecolor="whitesmoke")

sizes = 2 * 2 ** df["mag"]
# scatter's `c` needs plain numbers, not datetimes -- convert to
# matplotlib's numeric date representation, then format the colorbar
# ticks back into readable dates below.
time_num = mdates.date2num(df["time"])
vmin = mdates.date2num(pd.to_datetime("1980-01-01"))
vmax = mdates.date2num(pd.to_datetime("2026-07-31"))

sc = ax.scatter(df["longitude"], df["latitude"],
                 s=sizes, c=time_num, cmap="viridis",
                 alpha=0.4, edgecolor="k", linewidth=0.2,
                 transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
cbar.set_label("Date")
cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(cbar.ax.yaxis.get_major_locator()))

ax.set_title(f"Cascadia catalog: {len(df):,} events "
             "(color = date, size = magnitude)")
ax.gridlines(draw_labels=True, linewidth=0.3)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Wrote map to {OUT_PNG}")
plt.show()


#%%
# Plot 3
# Magnitude vs time -- stem plot
OUT_PNG = f"{DATA_DIR}/cascadia_magnitude_stem.png"
fig, ax = plt.subplots(figsize=(15, 6))

markerline, stemlines, baseline = ax.stem(df["time"], df["mag"], basefmt=" ")
markerline.set_marker("None")  # stem's markerline can't do per-point size/color -- draw markers via scatter below instead
plt.setp(stemlines, linewidth=0.4, alpha=0.5)

sizes = 2 * 2 ** df["mag"]
sc = ax.scatter(df["time"], df["mag"], s=sizes, c=df["mag"], cmap="viridis",
                 alpha=0.7, edgecolor="k", linewidth=0.2, zorder=3)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Magnitude")

ax.set_xlabel("Date")
ax.set_ylabel("Magnitude")
ax.set_title(f"Cascadia catalog: {len(df):,} events (magnitude vs. date)")
ax.grid(True, alpha=0.3)
ax.set_ylim([2,7.5])

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Wrote plot to {OUT_PNG}")
plt.show()