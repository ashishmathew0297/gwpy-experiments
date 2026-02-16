import numpy as np
import pandas as pd
from gwpy.table import (Table, EventTable)
from gwpy.segments import DataQualityFlag

def getEventList(file, start, end, margin=300):
    """
        Get event list between interval.
    Input
    -----
    file: list containing all events
    https://www.gw-openscience.org/eventapi/html/allevents/
    
    start: (float) start time of interval
    end: (float) end time of interval
    margin:  (int, optional) Margin around confirmed event. 
        The default is 300.
        
    Output
    ------
    events: all events in interval
    
    """
    df = pd.read_csv(file) # O1, O2, O3, O4
    df = df[(df['gps'] > start) & (df['gps'] < end)]
    eventlist = df['gps']
    events =list()
    for event in eventlist:
        if (event > start + margin) and event < end - margin:
            events.append(event)
    return events

def getSegments(all_dqf, events):
    """
        Generate numpy array from DQ flags
    Input
    -----
    all_dqf: (gwpy DataQuality) All DQ flags to transform
    
    Output
    ------
    segments: (numpy) array of start times and end times
    """
    starts, ends = list(), list()
    for e in range(len(all_dqf.active)):
        start = all_dqf.active[e][0]
        end = all_dqf.active[e][1]
        if events:
            starts.append(all_dqf.active[e][0])
            ends.append(all_dqf.active[e][1])
        else:
            if (end - start > 300):
                starts.append(all_dqf.active[e][0])
                ends.append(all_dqf.active[e][1])

    segments = np.vstack([np.asarray(starts), np.asarray(ends)])
    segments = np.transpose(segments)
    return segments

# Flags CBC-CAT1 and BURST-CAT1 of O3a are the same
def get_DQ_segments(ifo, start, end, run):
    flags = [f'{ifo}_DATA', f'{ifo}_CBC_CAT1', f'{ifo}_CBC_CAT2']

    eventlist = getEventList('/home/melissa.lopez/gwpy-experiments/dq_data/event-versions.csv', start, end) # O1, O2, O3, O4
    # create a DQF for all confident events and a 4 minute region around them
    event_dqf = ~DataQualityFlag(name='GW Events',
                             known=[(start, end)],
                             active=[(event-152, event+152) for event in eventlist])
    # Triple-coincidence + GW events from GWOSC
    all_dqf = event_dqf
    for flag in flags:
        dqf_open = DataQualityFlag.fetch_open_data(flag, start, end)
        all_dqf = all_dqf & dqf_open

    segments = getSegments(all_dqf, False)
    df_seg = pd.DataFrame(segments, columns=['start_time', 'end_time'])
    return df_seg
