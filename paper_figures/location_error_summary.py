# Summary table of MAP location errors (km) for the case-study location_hist figures:
# median / min / max over all M >= MIN_MAG events, plus the mainshock's own error,
# for Uniform, KDE and ETAS at TRIGGER_NUMBER triggers.
# Usage: python location_error_summary.py   (also writes location_error_summary.csv here)
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.dirname(os.path.abspath(__file__))
MAX_TRIGS, TRIGGER_NUMBER = 10, 4   # same as location_hist_case_study.py
MIN_MAG = float(os.environ.get('MIN_MAG', 3.5))   # events with ANSS magnitude >= this

CASES = {   # name: (mainshock id, catalog file)
    'MTJ 2024 M7': ('MTJ_2024_M7', 'nc75095651', 'mtj_2024_m7.csv'),
    'Ridgecrest':  ('Ridgecrest',  'ci38457511', 'Ridgecrest_2019_catalog.parquet'),
    'Ferndale':    ('Ferndale',    'nc73821036', 'Ferndale_2022_catalog.parquet'),
    'El Mayor':    ('ElMayor',     'ci14607652', 'El_Mayor-Cucapah_2010_catalog.parquet'),
}
PRIORS = {   # label: (subdir, csv)
    'Uniform': ('time_independent', 'uniform_benchmark_results.csv'),
    'KDE':     ('time_independent', 'kde_seismicity_benchmark_results.csv'),
    'ETAS':    ('time_dependent',   'etas_dynamic_benchmark_results.csv'),
}

rows = []
for label, (folder, main_id, cat_file) in CASES.items():
    cs = os.path.join(ROOT, 'results', 'california', 'case_studies', folder)
    cat_path = os.path.join(ROOT, 'data', 'california', 'case_studies', folder, cat_file)
    cat = pd.read_parquet(cat_path) if cat_path.endswith('.parquet') else pd.read_csv(cat_path)
    keep = set(cat.loc[cat['mag'] >= MIN_MAG, 'id'].astype(str))
    for prior, (sub, csv) in PRIORS.items():
        df = pd.read_csv(os.path.join(cs, 'output', sub, f'max_trigs_{MAX_TRIGS}', csv))
        df['event_id'] = df['event_id'].astype(str)
        df = df[df['event_id'].isin(keep) & (df['n_trigs'] == TRIGGER_NUMBER)].dropna(subset=['map_err_km'])
        err = df['map_err_km']
        main = df.loc[df['event_id'] == main_id, 'map_err_km']
        rows.append({'case': label, 'prior': prior, 'n_events': len(err),
                     'median_km': err.median(), 'min_km': err.min(), 'max_km': err.max(),
                     'mainshock_km': main.iloc[0] if len(main) else float('nan')})

table = pd.DataFrame(rows)
print(f'MAP location error (km) at {TRIGGER_NUMBER} triggers, events with M >= {MIN_MAG}')
print(table.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
table.to_csv(os.path.join(OUT, 'location_error_summary.csv'), index=False, float_format='%.2f')

# LaTeX table (booktabs).  The two magnitude-error columns are left blank until those results exist.
def cell(x):
    return '--' if pd.isna(x) else f'{x:.1f}'
blocks = []
for label in CASES:
    sub = table[table['case'] == label]
    blocks.append([[label if i == 0 else '', r.prior, str(r.n_events), cell(r.median_km), cell(r.min_km), cell(r.max_km),
                    cell(r.mainshock_km), '', ''] for i, r in enumerate(sub.itertuples())])
widths = [max(len(r[j]) for b in blocks for r in b) for j in range(9)]
def line(r):    # first two columns left-aligned, the rest right-aligned
    return '    ' + ' & '.join(r[j].ljust(widths[j]) if j < 2 else r[j].rjust(widths[j]) for j in range(9)) + r' \\'
body = '\n    \\midrule\n'.join('\n'.join(line(r) for r in b) for b in blocks)
tex = rf"""\begin{{table}}[htbp]
  \centering
  \small
  \setlength{{\tabcolsep}}{{4pt}}
  \caption{{Location errors for the case-study sequences with station masking, with $M \geq {MIN_MAG:g}$ (ANSS magnitude)}}
  \label{{tab:case_studies}}
  \begin{{tabular}}{{llrrrrrrr}}
    \toprule
    & & & \multicolumn{{4}}{{c}}{{Location error (km)}} & \multicolumn{{2}}{{c}}{{Magnitude error}} \\
    \cmidrule(lr){{4-7}} \cmidrule(lr){{8-9}}
    Case & Prior & $N$ & Median & Min & Max & Mainshock & Median & Max $|\cdot|$ \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
with open(os.path.join(OUT, 'location_error_summary.tex'), 'w') as f:
    f.write(tex)
print(tex)
