#%%
from priors import SeismicPrior
from bEPIC import EPIC_locate_prelim
import os
import pandas as pd

# Get root. Priors are in a different repo (priors/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get directory of THIS project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

data_dir = SeismicPrior.data_dir  # priors/data/

# Cached .tt3 paths for each prior
cache_paths = {
    'Gear1':             os.path.join(data_dir, 'GEAR1_prior.tt3'),
    'NSHM':              os.path.join(data_dir, 'USGS_NSHM_prior.tt3'),
    'Helmstetter':       os.path.join(data_dir, 'helmstetter_prior.tt3'),
    'Smooth_seismicity': os.path.join(data_dir, 'prior_seis_grid_US_Canada.tt3'),
    'ETAS':              os.path.join(data_dir, 'etas_prior.tt3'),  # set filename as needed
}

#%%
"""
# --- Step 1: Build and cache all priors ---
# Set construct = True to rebuild; False to skip if .tt3 files already exist.
"""
construct = True

if construct:
    try:
        p = SeismicPrior.from_gear1(os.path.join(data_dir, 'GEAR1_data', 'GL_HAZTBLT_M5_B2_2013.TMP'))
        p.to_tt3(cache_paths['Gear1'])
        print("Gear1: built and cached.")
    except Exception as e:
        print(f"Gear1: failed — {e}")

    try:
        p = SeismicPrior.from_nshm(os.path.join(data_dir, 'USGS_NSHM_data', 'gridded_moment_rates.xyz'))
        p.to_tt3(cache_paths['NSHM'])
        print("NSHM: built and cached.")
    except Exception as e:
        print(f"NSHM: failed — {e}")

    try:
        p = SeismicPrior.from_helmstetter()
        p.to_tt3(cache_paths['Helmstetter'])
        print("Helmstetter: built and cached.")
    except Exception as e:
        print(f"Helmstetter: failed — {e}")

    try:
        SeismicPrior.from_smooth_seismicity()  # validates the file loads cleanly
        print("Smooth_seismicity: ready (pre-built).")
    except Exception as e:
        print(f"Smooth_seismicity: failed — {e}")

    # TODO - incorporate calling this to build it, or have a dataframe or dict of saved options? 
    # Lots of ways to call this.
    # ETAS requires external inputs — build separately and save:
    #   p = SeismicPrior.from_etas(lats, lons, lambda_grid, forecast_time=t, metadata={...})
    #   p.to_tt3(cache_paths['ETAS'])
    print("ETAS: skipped (requires external ETAS output — build manually).")

#%%
"""
# --- Step 2: Select and load prior for bEPIC ---
# Options: 'Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS'
"""



#%%
# initial location, the rest are just dummy variables
# """
# Dummy event
# """
# event = EPIC_locate_prelim.Event(lat = 36.764, 
#               lon = -121.4472, 
#               time = 1538771380.11, 
#               misfit_rms = 0, 
#               misfit_ave = 0, 
#               eventid = 126625, 
#               version = 0)

# ''' 126625.run
# version,order,station,channel,network,location,longitude,latitude,trigger time,tterr,logPd
# 0,1,SAO,HNZ,BK,00,-121.4472,36.764,1538771380.09,-0.053,-1.89848
# 0,2,SAO,HHZ,BK,00,-121.4472,36.764,1538771380.11,-0.073,-1.875756
# 0,3,BSR,HNZ,NC,--,-121.5203,36.6674,1538771381.18,-0.035,-2.964038
# 0,5,PACP,HHZ,BK,00,-121.287,37.008,1538771383.34,0.081,-2.38656
# 0,4,PACP,HNZ,BK,00,-121.287,37.008,1538771383.34,0.081,-2.428661
# '''
# t = EPIC_locate_prelim.TriggerManager(lon = -121.4472, lat = 36.764, sta='SAO', net='BK', chan='HNZ',trigger_time = 1538771380.09)
# event.trigs.append(t)

# t = EPIC_locate_prelim.TriggerManager(lon = -121.4472, lat = 36.764, sta='SAO', net='BK', chan='HHZ',trigger_time = 1538771380.11)
# event.trigs.append(t)

# t = EPIC_locate_prelim.TriggerManager(lon = -121.5203, lat = 36.6674, sta='BSR', net='NC', chan='HNZ',trigger_time =1538771381.18)
# event.trigs.append(t)

# t = EPIC_locate_prelim.TriggerManager(lon =-121.287, lat = 37.008, sta='PACP', net='BK', chan='HHZ',trigger_time =1538771383.34)
# event.trigs.append(t)

# t = EPIC_locate_prelim.TriggerManager(lon =-121.287, lat = 37.008, sta='PACP', net='BK', chan='HNZ',trigger_time =1538771383.34)
# event.trigs.append(t)

# t,output_df = EPIC_locate_prelim.E2Location_locate(params,event)

#%%
# --- Real event from run file ---

# Get prior
selected = 'NSHM'
p = SeismicPrior.from_tt3(cache_paths[selected])

# Pass prior to param
params = EPIC_locate_prelim.EPIC_PARAMS()
params.prior = p
params.use_prior = True # Use the prior?
params.GridSize = 25 # Number of grid points or slices
params.GridKm = 50 # Total size of grid (in km, one dimension)
params.method = 'EPIC C'  
params.MAX_EVENT_TRIGS = 25 # Maximum amount of triggers to use
# Grid spacing = gridkm/gridsize

event_id = 126625
version  = None
run_path = os.path.join(PROJECT_ROOT, 'run_files', f'{event_id}.run')

df_run = pd.read_csv(run_path)

df_run = df_run.rename(columns={'trigger time': 'trigger_time'})
first = df_run.sort_values('order').iloc[0]

# Create initial event
event = EPIC_locate_prelim.Event(
    lat        = first['latitude'],
    lon        = first['longitude'],
    time       = first['trigger_time'],
    misfit_rms = 0,
    misfit_ave = 0,
    eventid    = event_id,
    version    = 0,
)
results = {}

# Iterate over the versions (new version every time new trigger)
for version in sorted(df_run['version'].unique()):
    df_v = (df_run[df_run['version'] == version]
            .sort_values('order')
            .head(params.MAX_EVENT_TRIGS))

    event.trigs = []
    event.version = int(version)
    for row in df_v.itertuples(index=False):
        trig = EPIC_locate_prelim.TriggerManager(
            lon          = row.longitude,
            lat          = row.latitude,
            trigger_time = row.trigger_time,
            sta          = row.station,
            net          = row.network,
            chan         = row.channel,
        )
        event.trigs.append(trig)

    t, output_df = EPIC_locate_prelim.E2Location_locate(params, event)
    results[(event_id, version)] = (t, output_df)

    if len(df_v) >= params.MAX_EVENT_TRIGS:
        break


#%%

