# Summary table of MAP location errors (km) for the case-study location_hist figures:
# median / min / max over all M >= MIN_MAG events, plus the mainshock's own error,
# for Uniform, KDE and ETAS at TRIGGER_NUMBER triggers.
# Usage: python location_error_summary.py   (also writes location_error_summary.csv here)
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.dirname(os.path.abspath(__file__))
MAX_TRIGS, TRIGGER_NUMBER, MIN_MAG = 10, 4, 4.0   # same as location_hist_case_study.py

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
