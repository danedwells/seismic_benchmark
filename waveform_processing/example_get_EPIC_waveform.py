

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

network = 'CI'
name    = 'SGL'
channel = 'HNZ'
location = '--'
gain = 214168.50006901292 / 100 # because I have to convert from m to cm
trigger_time = '2020-08-28T11:26:05.558'

# getting gain: https://ds.iris.edu/mda/CI/SGL/

# -------------------------------------------------------------------------- #

starttime = UTCDateTime(trigger_time)  - 180
endtime   = UTCDateTime(trigger_time)  + 60

# get the trace via obspy
tr = process_EPIC_waveforms.get_trace_t(network, name, location, channel, starttime, endtime)

(data_a,data_v,data_d) = process_EPIC_waveforms.ToGroundTSPMoudle_EPIC(tr,gain)



#  ---------------------   plotting
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

# the filtering has a big edge effect. So get more data than you need and null out the beginnig
data_a[0:5000]=np.nan
data_v[0:5000]=np.nan
data_d[0:5000]=np.nan



fig,ax = plt.subplots(figsize=(10,4))

ax.plot(tr.times('matplotlib'),data_d,c='k')
ax.axvline(x = mdates.date2num(UTCDateTime(trigger_time)),c='r')

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.set_ylabel('disp (cm)')
ax.set_xlim([mdates.date2num(UTCDateTime(trigger_time)-5), mdates.date2num(UTCDateTime(trigger_time)+10)])



