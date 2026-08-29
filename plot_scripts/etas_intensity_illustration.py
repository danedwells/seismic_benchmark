"""
etas_intensity_illustration.py — Presentation figure: ETAS conditional
intensity immediately before vs. immediately after a new earthquake.

Illustrates the "bullseye" of triggered aftershock density that appears at
the epicenter the instant a new event occurs, decaying with distance
(the ETAS spatial kernel) and with time (Omori-Utsu). Uses a synthetic
toy catalog — no dependency on real data or the etas_2/priors packages —
so it runs standalone and is easy to retune for a talk.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#%%
# ---------------------------------------------------------------------------
# Config — tune these to change how the figure looks
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)

DOMAIN_KM = 40      # map extent, +/- km from center
GRID_N    = 300      # grid resolution per axis
N_HISTORY = 22       # number of historical background events
MU0       = 0.02     # constant background rate (events / km^2 / day)

# Omori-Utsu time decay: g(dt) = 1 / (dt + c)^p
C_OMORI, P_OMORI = 0.05, 1.1

# ETAS spatial kernel: f(r) = (q-1) / (pi * d^2) * (1 + (r/d)^2)^-q
# d grows with magnitude: d = D0 * 10^(ALPHA_D * (M - MC))
Q_KERNEL, D0, ALPHA_D = 1.6, 0.5, 0.35
MC = 2.5              # completeness magnitude
K_PROD = 0.05         # productivity scale

MAINSHOCK_MAG = 5.5
MAINSHOCK_LOC = (0.0, 0.0)

OUT_PATH = 'etas_intensity_illustration.png'

# ---------------------------------------------------------------------------
# Synthetic catalog: scattered historical events over the past 30 days,
# plus one new mainshock at t = 0.
# ---------------------------------------------------------------------------
hist_x = rng.uniform(-DOMAIN_KM * 0.8, DOMAIN_KM * 0.8, N_HISTORY)
hist_y = rng.uniform(-DOMAIN_KM * 0.8, DOMAIN_KM * 0.8, N_HISTORY)
hist_m = rng.uniform(MC, 4.3, N_HISTORY)
hist_t = -rng.uniform(0.5, 30, N_HISTORY)   # days before the mainshock

# ---------------------------------------------------------------------------
# ETAS field
# ---------------------------------------------------------------------------
def omori(dt):
    return 1.0 / (dt + C_OMORI) ** P_OMORI

def spatial_kernel(r, mag):
    d = D0 * 10 ** (ALPHA_D * (mag - MC))
    return (Q_KERNEL - 1) / (np.pi * d ** 2) * (1 + (r / d) ** 2) ** (-Q_KERNEL)

def conditional_intensity(xx, yy, t, events_x, events_y, events_m, events_t):
    lam = np.full_like(xx, MU0)
    for ex, ey, em, et in zip(events_x, events_y, events_m, events_t):
        if et >= t:
            continue
        r = np.hypot(xx - ex, yy - ey)
        lam += K_PROD * 10 ** (em - MC) * spatial_kernel(r, em) * omori(t - et)
    return lam

grid = np.linspace(-DOMAIN_KM, DOMAIN_KM, GRID_N)
xx, yy = np.meshgrid(grid, grid)

t_before, t_after = -1e-3, 1e-3   # an instant before / after the mainshock

lam_before = conditional_intensity(xx, yy, t_before, hist_x, hist_y, hist_m, hist_t)

all_x = np.append(hist_x, MAINSHOCK_LOC[0])
all_y = np.append(hist_y, MAINSHOCK_LOC[1])
all_m = np.append(hist_m, MAINSHOCK_MAG)
all_t = np.append(hist_t, 0.0)
lam_after = conditional_intensity(xx, yy, t_after, all_x, all_y, all_m, all_t)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True,
                          sharex=True, sharey=True)

norm = LogNorm(vmin=MU0 / 2, vmax=lam_after.max())
cmap = plt.get_cmap('viridis')

axes[0].pcolormesh(xx, yy, lam_before, norm=norm, cmap=cmap, shading='auto')
axes[0].set_title('Before: background seismicity only', fontsize=12)

im1 = axes[1].pcolormesh(xx, yy, lam_after, norm=norm, cmap=cmap, shading='auto')
axes[1].set_title('After: new earthquake ignites the aftershock kernel', fontsize=12)

for ax in axes:
    ax.scatter(hist_x, hist_y, s=15 + 8 * (hist_m - MC), facecolor='none',
               edgecolor='white', linewidth=0.8, label='Historical events')
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (km)', fontsize=11)

axes[0].set_ylabel('Northing (km)', fontsize=11)

axes[0].scatter(*MAINSHOCK_LOC, marker='*', s=220, facecolor='none',
                edgecolor='red', linewidth=1.5, linestyle='--',
                label='Upcoming mainshock')
axes[1].scatter(*MAINSHOCK_LOC, marker='*', s=220, color='red',
                edgecolor='black', linewidth=0.8,
                label=f'M{MAINSHOCK_MAG} mainshock')

axes[0].legend(loc='upper left', fontsize=8, framealpha=0.85)
axes[1].legend(loc='upper left', fontsize=8, framealpha=0.85)

cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label(r'Conditional intensity $\lambda(x,y,t)$  [events / km$^2$ / day]',
               fontsize=11)

fig.suptitle('ETAS conditional intensity: before vs. after a new earthquake',
             fontsize=14)

fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
plt.show()

# %%
