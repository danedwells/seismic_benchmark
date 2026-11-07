"""
LaTeX table of location errors for the EPIC replay logs, one block per case-study sequence and one row per run
(Control, Sigma Only, Sigma and ETAS).  Same layout as the paper's case-study table; sequences without log results
yet (no log_results_<SEQUENCE>_nS<N>.csv from run_error_histograms_comp.py) are filled with '--'.

Run run_error_histograms_comp.py once per sequence first (SEQUENCE=MTJ, SEQUENCE=Ridgecrest, ...) at the same N_TRIGS.
Writes table_case_studies_logs.tex next to this script and prints it.
"""
import os
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import benchmark.config_california as config

N_TRIGS = int(os.environ.get('N_TRIGS', 4))        # must match the N_TRIGS the results were made with
CASES = [('MTJ 2024 M7', 'MTJ', 'MTJ_2024_M7'), ('Ridgecrest', 'Ridgecrest', 'Ridgecrest'),
         ('Ferndale', 'Ferndale', 'Ferndale'), ('El Mayor', 'ElMayor', 'ElMayor')]       # (table name, SEQUENCE, config key)
RUNS = {'control': 'Control', 'sigma_only': 'Sigma Only', 'sigma_and_prior': 'Sigma and ETAS'}
MISSING = ['--'] * 7

def tex_num(x):
    """Two decimals; a negative number gets a true minus sign (math mode)."""
    return f'${x:.2f}$' if x < 0 else f'{x:.2f}'

blocks = []
for name, seq, key in CASES:
    path = HERE / f'log_results_{seq}_nS{N_TRIGS}.csv'
    df = pd.read_csv(path) if path.exists() else None
    rows = []
    for i, (run, label) in enumerate(RUNS.items()):
        cells = MISSING
        if df is not None:
            r = df[df.run == run]
            ms = r[r.anss_id == config.FOCUS_EVENTS_MAINSHOCK[key]].loc_err_km
            cells = [str(len(r)), f'{r.loc_err_km.median():.1f}', f'{r.loc_err_km.min():.1f}', f'{r.loc_err_km.max():.1f}',
                     f'{ms.iloc[0]:.1f}' if len(ms) else '--',
                     tex_num(r.mag_err.median()), tex_num(r.mag_err.abs().max())]
        rows.append([name if i == 0 else '', label] + cells)
    blocks.append(rows)

widths = [max(len(r[j]) for b in blocks for r in b) for j in range(9)]
def line(r):                                         # first two columns left-aligned, numbers right-aligned (as in the booktabs layout)
    cells = [r[j].ljust(widths[j]) if j < 2 else r[j].rjust(widths[j]) for j in range(9)]
    return '    ' + ' & '.join(cells) + r' \\'
body = ('\n    \\midrule\n').join('\n'.join(line(r) for r in b) for b in blocks)

tex = rf"""% EPIC replay logs: first alerted version of the matched EPIC event with >= {N_TRIGS} stations, USGS M >= 3.5; Mainshock = that event's error.
\begin{{table}}[htbp]
  \centering
  \small
  \setlength{{\tabcolsep}}{{4pt}}
  \caption{{Location errors for the case-study sequences. \todo{{Caption.}}}}
  \label{{tab:case_studies_logs}}
  \begin{{tabular}}{{llrrrrrrr}}
    \toprule
    & & & \multicolumn{{4}}{{c}}{{Location error (km)}} & \multicolumn{{2}}{{c}}{{Magnitude error}} \\
    \cmidrule(lr){{4-7}} \cmidrule(lr){{8-9}}
    Case & Run & $N$ & Median & Min & Max & Mainshock & Median & Max $|\cdot|$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
(HERE / 'table_case_studies_logs.tex').write_text(tex)
print(tex)
