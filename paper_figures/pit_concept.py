#%%
# =============================================================================
# pit_concept.py — conceptual (dummy-data) cartoon: why the credible level at the
# true location is a CDF value, and therefore a valid probability integral transform.
# =============================================================================
# Left:  a 1D Gaussian posterior. The horizontal line is the posterior density at the
#        true (USGS) location; the mass of the region ABOVE that line (the highest-density
#        region that just includes the truth) is the credible level.
# Right: the same posterior as N samples, sorted by their density value. Same y-axis,
#        so the horizontal line crosses the curve at the cumulative probability of the
#        mass below it; the credible level is 1 minus that. It is a CDF value of the
#        posterior's own density, so if the posterior is calibrated it is uniform on (0, 1).
# Dummy numbers only.
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MU, SIGMA = 0.0, 10.0       # posterior mean / std (km)
X_TRUE    = 6.0             # "USGS" location (km)
N_SAMPLES = 30              # samples drawn for the sorted panel

ABOVE, BELOW = '#1f77b4', '#bbbbbb'

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.labelsize': 12, 'figure.dpi': 150})

p_true = norm.pdf(X_TRUE, MU, SIGMA)                      # density at the true location
level  = 2 * norm.cdf(abs(X_TRUE - MU) / SIGMA) - 1       # mass above the line = credible level
x = np.linspace(-32, 32, 600)
pdf = norm.pdf(x, MU, SIGMA)
x_hdr = abs(X_TRUE - MU)                                  # density >= p_true  <=>  |x - mu| <= x_hdr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, gridspec_kw={'wspace': 0.08})

# -- Left: density, shaded by whether it is above / below the true location's density --
ax1.plot(x, pdf, color='k', lw=2)
ax1.fill_between(x, pdf, where=np.abs(x - MU) <= x_hdr, color=ABOVE, alpha=0.6)
ax1.fill_between(x, pdf, where=np.abs(x - MU) >= x_hdr, color=BELOW, alpha=0.7)
ax1.axhline(p_true, color='crimson', ls='--', lw=2)
ax1.plot([X_TRUE], [p_true], 'o', color='crimson', ms=9, zorder=5)
ax1.text(X_TRUE + 1.2, p_true + 0.0015, 'true (USGS)\nlocation', color='crimson', ha='left', va='bottom',
         fontsize=12, fontweight='bold')
ax1.text(MU, p_true / 2, f'mass above\n= {level:.2f}', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')
ax1.annotate(f'mass below\n= {1 - level:.2f}', xy=(-17, 0.004), xytext=(-20, 0.030), ha='center', va='center',
             fontsize=12, fontweight='bold', color='#444444',
             arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
ax1.set_xlabel('Position (km)')
ax1.set_ylabel('Posterior density')
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(0, 0.043)
ax1.set_yticks([])

# -- Right: samples sorted by density -> CDF of the density value --------------
rng = np.random.default_rng(11)
dens = np.sort(norm.pdf(rng.normal(MU, SIGMA, N_SAMPLES), MU, SIGMA))
ranks = np.arange(1, N_SAMPLES + 1) / N_SAMPLES
ax2.step(ranks, dens, where='post', color=ABOVE, lw=2, label=f'samples sorted by density\n(rank / N, N = {N_SAMPLES})')
ax2.plot(ranks, dens, 'o', color=ABOVE, ms=4)
d = np.linspace(1e-6, norm.pdf(0, MU, SIGMA), 400)
z = np.sqrt(-2 * np.log(d * SIGMA * np.sqrt(2 * np.pi)))
ax2.plot(2 * (1 - norm.cdf(z)), d, color='k', lw=1.2, alpha=0.6, label='true CDF of the density value')
ax2.axhline(p_true, color='crimson', ls='--', lw=2)
ax2.plot([1 - level], [p_true], 'o', color='crimson', ms=9, zorder=5)
ax2.plot([1 - level, 1 - level], [0, p_true], color='crimson', ls=':', lw=2)
ax2.text(1 - level - 0.02, 0.002, f'{1 - level:.2f}', color='crimson', ha='right', va='bottom',
         fontsize=13, fontweight='bold')
ax2.text(0.98, 0.004, f'credible level\n= 1 − {1 - level:.2f}\n= {level:.2f}', color='crimson', ha='right',
         va='bottom', fontsize=13, fontweight='bold')
ax2.set_xlabel('Cumulative probability')
ax2.set_xlim(0, 1)
ax2.legend(loc='upper left', fontsize=10, frameon=False, bbox_to_anchor=(0.0, 1.0))
for ax in (ax1, ax2):
    ax.spines[['top', 'right']].set_visible(False)

fig.savefig(os.path.join(OUTPUT_DIR, 'pit_concept.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'pit_concept.pdf'), bbox_inches='tight')
plt.show()
