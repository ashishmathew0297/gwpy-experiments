import pandas as pd
from dq_utils import get_DQ_segments
from gwpy.segments import DataQualityFlag

run = 'O3a'
scratch = 10 # just to check their surroundings
ifo = 'L1'
start = 1238166018 #O3a start
end = 1253977218 #O3a end

path_glitches = f'../GPStimes/{run}_allifo.csv' # FIXME
glitches = pd.read_csv(path_glitches)
glitches = glitches[glitches['ifo'] == ifo]
print(len(glitches))

segments = get_DQ_segments(ifo, start, end)

# Example DataFrame 2 (time points to check)
df2 = glitches.copy()


# Check if each row in df2 falls within any interval in df1
mask = glitches.apply(lambda row: ((segments['start_time'] <= row['GPStime']) - scratch & (segments['end_time'] >= row['GPStime'] + scratch)).any(), axis=1)

# Filter df2 to get only matching rows
DQ_glitches = glitches[mask]
print(len(DQ_glitches))

DQ_glitches.to_csv(f'DQ_glitches_{run}.csv')