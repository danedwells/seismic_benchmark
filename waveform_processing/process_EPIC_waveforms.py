
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: amy
"""

import pandas as pd
import numpy as np
from obspy import UTCDateTime


def build_filter(iord,filter_type,fl,fh,time_interval,zp):
  
    tdel = time_interval
    
    # butterPoles
    ptype = np.tile('',10)
    p = np.empty(10).astype(complex)
    
    p,ptype,nps = butterPoles(p, ptype, iord)
    
    sn = np.zeros(nps*6)
    sd = np.zeros(nps*6)
    
    # high pass
    fl = fl*tdel/2.;
    flw = tangent_warp( fl, 2. );
    
    (sn,sd,nsects) = LPtoHP(p, ptype, nps, sn,sd);
    (sn,sd       ) = cutoffAlter(flw,nsects,sn,sd)
    (sn, sd      ) = bilinear(nsects,sn,sd)

    return(sn,sd,nsects)

def applyFilter(filter_params, data, data_length):
    
    sn = filter_params[0]
    sd = filter_params[1]
    nsects = filter_params[2]

    nsamps = data_length
  
    x1 = 0;   x2 = 0;   y1 = 0;   y2 = 0
    for i in range(nsamps):
        
        jptr = 0 
        input_val  = data[i]
        output_val = input_val
        
        for j in range(int(nsects)):
            
            output_val = sn[jptr] * input_val + sn[jptr+1] * x1 + sn[jptr+2] * x2    - ( sd[jptr+1] * y1 + sd[jptr+2] * y2 )
            
            y2 = y1
            y1 = output_val
            x2 = x1
            x1 = input_val
            
            jptr += 3
            input_val = output_val
        
        data[i] = output_val
    
   
    return(data)

def mathfun_integrate(x,y,nsamps,prevx,prevy,dt2):
    y[0] = (x[0] + prevx)*dt2 + prevy
    for i in range(1,nsamps):
        y[i] = (x[i] + x[i-1]) * dt2 + y[i-1]
    prevy = y[nsamps-1]
    
    return(y,prevy)

def mathfun_differentiate(x,y,nsamps,prevx,dt):
    y[0] = (x[0] - prevx)/dt;
    for i in range(1,nsamps):
        y[i] = (x[i] - x[i-1])/dt;
        prevx = x[i];
        
    return(y,prevx)

def butterPoles(p,ptype,iord):
    half = (iord/2)
    n=0
    for k in range(int(half)):
        
        angle = np.pi * (0.5 + (2 *(k +1) -1) / (2*iord ) )
        p[n]     = complex( np.cos( angle), np.sin(angle) )
        ptype[n] = 'C'
        n+=1
    
    return(p,ptype,n)


def LPtoHP(p,ptype,nps,sn,sd):
    nsects = 0
    for i in range(nps):
        iptr = 0
        if ptype[i] == 'C':
            sn[ iptr ]      = 0
            sn[ iptr + 1 ]  = 0
            sn[ iptr + 2 ]  = 1
            sd[ iptr ]      = 1
            sd[ iptr + 1 ]  = -2 * p[i].real
            sd[ iptr + 2 ]  = np.multiply(p[i],np.conjugate(p[i])).real
           
            iptr += 3
            nsects +=1
    return(sn,sd, nsects)

def tangent_warp(f,t):
    twopi = 2*np.pi  
    fac = .5 * f *t
    if fac >= 0.25: fac = .2499999
    angle = fac*twopi
    
    
    warp = 2*np.tan(angle )/t
    warp = warp/twopi

    return(warp)

def cutoffAlter(f,nsects,sn,sd):
    scale = 2 * np.pi * f
    for i in range(nsects):
        iptr = 0
        
        sn[ iptr + 1] = sn[ iptr + 1] / scale
        sn[ iptr + 2] = sn[ iptr + 2] / (scale*scale)
        sd[ iptr + 1] = sd[ iptr + 1] / scale
        sd[ iptr + 2] = sd[ iptr + 2] / (scale*scale)
        iptr += 3
    return(sn,sd)

def bilinear(nsects,sn,sd):

    for i in range(nsects):
        iptr = 0
        
        a0 = sd[iptr];
        a1 = sd[iptr+1];
        a2 = sd[iptr+2];
        
        scale = a2 + a1 + a0;
        sd[iptr]   = 1.;
        sd[iptr+1] = (2.*(a0 - a2)) / scale;
        sd[iptr+2] = (a2 - a1 + a0) / scale;
        
        a0 = sn[iptr];
        a1 = sn[iptr+1];
        a2 = sn[iptr+2];
        
        sn[iptr]   = (a2 + a1 + a0) / scale;
        sn[iptr+1] = (2.*(a0 - a2)) / scale;
        sn[iptr+2] = (a2 - a1 + a0) / scale;
        iptr = iptr + 3;
        
    return(sn, sd)


def _fdsn_client(network):
    from obspy.clients.fdsn import Client
    if   network == 'CI':           return Client("SCEDC")
    elif network in ('BK', 'NC'):   return Client("NCEDC")
    else:                           return Client("IRIS")


def get_trace_t(network, station, location, channel, starttime, endtime):
    client = _fdsn_client(network)
    st = client.get_waveforms(network, station, location, channel, starttime, endtime)
    return st[0]


def get_trace_with_metadata(network, station, location, channel, starttime, endtime):
    """
    Fetch waveform, gain (cm units), and station coordinates in one call.

    Returns
    -------
    tr       : obspy Trace
    gain_cm  : float  — instrument sensitivity in DU/(cm/s) or DU/(cm/s²)
    sta_lat  : float
    sta_lon  : float
    """
    client = _fdsn_client(network)
    st  = client.get_waveforms(network, station, location, channel, starttime, endtime)
    inv = client.get_stations(network=network, station=station,
                              location=location, channel=channel,
                              starttime=starttime, endtime=endtime,
                              level='response')
    tr  = st[0]
    ch  = inv[0][0][0]
    gain_cm = ch.response.instrument_sensitivity.value / 100  # DU/m → DU/cm
    return tr, gain_cm, ch.latitude, ch.longitude

def ToGroundTSPMoudle_EPIC(tr,gain):
    import numpy as np
    from datetime import datetime
    
    '''
    Last updated: April 7 2025
    
    converts raw trace to displacement, velocity, and acceleration the same
    way as EPICWP.
    
    inputs:
        obspy trace(tr) -  which must contain npts and sampling_rate values.
        gain value (G)  - the units of this must be in cm to match with EPIC
        
        so check the gain/sensitivity , if units =='DU/M/S**2', then G=G/100
                                             OR
                                        if units =='DU/M/S',    then  G=G/100  
     
    outputs:
        acceleration 
        velocity
        displacement
        time (from the trace)
    

    '''
    
    data_raw         = tr.data 
    nsamps           = int(tr.stats.npts)
    chan_samprate    = int(tr.stats.sampling_rate)
    delta_t          = 1.0/chan_samprate;
    
    # these are values set in the EPIC_WP.conf file.  
    para_filt_hpfc   = 0.075    # HPFreqCutoff
    para_filt_order  = 2        # HPFilterOrder
    para_blwin       = 60       # BaselineWin
      
    
    
    waveform_type = 'velocity'
    if tr.stats.channel[1] == 'N': waveform_type = 'acceleration'
    if tr.stats.channel[1] == 'L': waveform_type = 'acceleration'
    if tr.stats.channel[1] == 'H': waveform_type = 'velocity'
    
    var_a0   = 0.0;    var_vuf0 = 0.0
    var_z0   = 0.0;    var_duf0 = 0.0
    var_v0   = 0.0;    
    running_sum=0
   
    # // Integrate/differentiate
    dt2 = delta_t/2.0

    # set up array sizes
    array_size = nsamps
    r      = np.zeros(array_size)*np.nan;   
    z      = np.zeros(array_size)*np.nan
    data_a = np.zeros(array_size)*np.nan;   
    data_v = np.zeros(array_size)*np.nan
    data_d = np.zeros(array_size)*np.nan
   
    
    
    # build filter
    riir = build_filter(para_filt_order,'HP',para_filt_hpfc,0,delta_t,0)


    # ---------------------- Baseline removal ---------------------- 
    max_samples = para_blwin * chan_samprate;
    bl_array    = np.zeros(int(max_samples))   # C++ is allocating space for bl_array

    bl_index = 0 ;     bl_size = 0
    #// Array baseline correction
    for i in range(nsamps):
        running_sum -= bl_array[bl_index]
        running_sum += data_raw[i]
        
        if (bl_size < max_samples): bl_size +=1
        bl_array[bl_index] = data_raw[i]
        bl_size +=1
        if (bl_index == max_samples): bl_index = 0
        
        long_term_av = running_sum / bl_size;

        z[i] = (data_raw[i]-long_term_av)/gain     # Z is gain corrected and long term average removed
      
        if(waveform_type == 'acceleration'):    data_a[i] = z[i]
        else:                                   data_v[i] = z[i]



    # -------------------------------- filtering ---------------------- 

    if(waveform_type == 'acceleration'): 
        #// 2-pole filter raw -> acc
        
        data_a = applyFilter(riir, data_a, nsamps)
        
        #// integration acc -> vel 
        data_v, var_vuf0=mathfun_integrate(data_a,data_v,nsamps,var_a0, var_vuf0,dt2)

    else:
        #// differentiate vel -> acc
        data_a, var_z0 = mathfun_differentiate(data_v,data_a,nsamps,var_z0,delta_t)
        data_a = applyFilter(riir, data_a, nsamps)
    
    var_a0 = data_a[nsamps-1];
    data_v = applyFilter(riir, data_v, nsamps)

    #// integrate vel -> disp
    data_d, var_duf0=mathfun_integrate(data_v,data_d,nsamps,var_v0, var_duf0,dt2)

    #// 2-pole filter disp
    data_d = applyFilter(riir, data_d, nsamps)
    
    return(data_a,data_v,data_d)


def get_station_latlon(network, station, location, channel, time):
    """Fetch station latitude and longitude from FDSN at a given time."""
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    if   network == 'CI':             client = Client("SCEDC")
    elif network in ('BK', 'NC'):     client = Client("NCEDC")
    else:                             client = Client("IRIS")

    t = UTCDateTime(time)
    inv = client.get_stations(network=network, station=station,
                              location=location, channel=channel,
                              starttime=t, endtime=t + 86400,
                              level='channel')
    cha = inv[0][0][0]
    return cha.latitude, cha.longitude


def hypocentral_dist_km(event_lat, event_lon, event_depth_km,
                        station_lat, station_lon):
    """Hypocentral distance in km."""
    from obspy.geodetics import gps2dist_azimuth
    epi_m, _, _ = gps2dist_azimuth(event_lat, event_lon, station_lat, station_lon)
    return np.sqrt((epi_m / 1000.0) ** 2 + event_depth_km ** 2)


def pd_magnitude(data_d, sampling_rate, trigger_sample, dist_km,
                 window_s=4.0, coeffs=None):
    """
    Single-station Pd magnitude from peak displacement in a window after P arrival.

    Relation (bEPIC paper):  M = c1*log10(Pd_cm) + c2*log10(R_km) + c3

    Default coefficients: Williamson et al., 2023

    The filter edge effect corrupts the first ~50 s of data_d (at 100 sps).
    Ensure trigger_sample is well past that region (fetch at least 3 min before
    the trigger, as shown in example_get_EPIC_waveform.py).

    Parameters
    ----------
    data_d        : displacement array in cm (output of ToGroundTSPMoudle_EPIC)
    sampling_rate : samples per second
    trigger_sample: sample index of P-wave arrival
    dist_km       : hypocentral distance in km (from hypocentral_dist_km)
    window_s      : seconds after trigger to search for peak (default 4.0)
    coeffs        : dict with keys c1, c2, c3 — override default scaling relation

    Returns
    -------
    M : float, or np.nan if the window is empty or contains only NaNs
    """
    if coeffs is None:
        coeffs = dict(c1=1.23, c2=1.39, c3=5.39)

    end_sample = int(trigger_sample + window_s * sampling_rate)
    window = data_d[trigger_sample:end_sample]

    if len(window) == 0 or np.all(np.isnan(window)):
        return np.nan

    Pd = np.nanmax(np.abs(window))
    if Pd <= 0:
        return np.nan

    c1, c2, c3 = coeffs['c1'], coeffs['c2'], coeffs['c3']
    return c1 * np.log10(Pd) + c2 * np.log10(dist_km) + c3
