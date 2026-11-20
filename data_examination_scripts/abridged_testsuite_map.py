#%%
# One-off: map of Amy Williamson's abridged test-suite events, coloured and sized by magnitude.
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "abridged_testsuite.csv").sort_values("magnitude")

fig = plt.figure(figsize=(9, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
pad = 1.5
ax.set_extent([df.lon.min() - pad, df.lon.max() + pad, df.lat.min() - pad, df.lat.max() + pad])
ax.add_feature(cfeature.LAND, facecolor="0.95")
ax.add_feature(cfeature.OCEAN, facecolor="0.88")
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="0.5")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
sc = ax.scatter(df.lon, df.lat, c=df.magnitude, s=(df.magnitude - 3) ** 3 * 8 + 10,
                cmap="plasma", edgecolor="k", linewidth=0.4, alpha=0.85,
                transform=ccrs.PlateCarree())
fig.colorbar(sc, ax=ax, shrink=0.6, label="Magnitude")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
gl.top_labels = gl.right_labels = False
ax.set_title(f"Abridged test suite: {len(df)} events, M {df.magnitude.min():.1f}–{df.magnitude.max():.1f}")
plt.tight_layout()
plt.show()

# %%
