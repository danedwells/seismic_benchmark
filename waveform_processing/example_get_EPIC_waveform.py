
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:14:15 2026

@author: amy
"""


import pandas as pd
import numpy  as np

from   obspy                     import read,Stream,UTCDateTime
from   obspy.clients.fdsn        import Client


import process_EPIC_waveforms

network      = 'CI'
name         = 'SGL'
channel      = 'HNZ'
location     = '--'
trigger_time = '2020-08-28T11:26:05.558'

# -------------------------------------------------------------------------- #

starttime = UTCDateTime(trigger_time) - 180
endtime   = UTCDateTime(trigger_time) + 60

tr, gain_cm, sta_lat, sta_lon = process_EPIC_waveforms.get_trace_with_metadata(
    network, name, location, channel, starttime, endtime)

(data_a, data_v, data_d) = process_EPIC_waveforms.ToGroundTSPMoudle_EPIC(tr, gain_cm)



#  ---------------------   plotting
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

# the filtering has a big edge effect. So get more data than you need and null out the beginnig
data_a[0:5000]=np.nan
data_v[0:5000]=np.nan
data_d[0:5000]=np.nan


#%%
fig,ax = plt.subplots(figsize=(10,4))

ax.plot(tr.times('matplotlib'),data_d,c='k')
ax.axvline(x = mdates.date2num(UTCDateTime(trigger_time)),c='r')

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.set_ylabel('disp (cm)')
ax.set_xlim([mdates.date2num(UTCDateTime(trigger_time)-5), mdates.date2num(UTCDateTime(trigger_time)+10)])



# -------------------------------------------------------------------------- #
#  Pd magnitude example
#  Event: M 4.6  2020-08-28  Fontana, CA
#  USGS hypocenter used for distance calculation
# -------------------------------------------------------------------------- #

event_lat   =  34.066
event_lon   = -117.504
event_depth =  10.0    # km

R_km = process_EPIC_waveforms.hypocentral_dist_km(
    event_lat, event_lon, event_depth, sta_lat, sta_lon)

# sample index of P arrival in the trace
trigger_sample = int((UTCDateTime(trigger_time) - starttime) * tr.stats.sampling_rate)

coeffs = dict(c1=1.23, c2=1.39, c3=5.39)
M_pd = process_EPIC_waveforms.pd_magnitude(
    data_d, tr.stats.sampling_rate, trigger_sample, R_km,
    window_s=4.0, coeffs = coeffs)

print(f'Station {name}  R={R_km:.1f} km  Pd-magnitude={M_pd:.2f}')

#%%