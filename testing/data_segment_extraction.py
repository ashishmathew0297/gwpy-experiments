import os
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pycbc
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch


def get_TimeSeries_data(gps_start_time: float, gps_end_time: float, srate=4096, ifo='L1') -> list:

    filepath = "../timeseries_data/"
    filename = f"sample_{ifo}_{gps_start_time}_{gps_end_time}.h5"
    
    # Create the directory if it does not exist
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    

    print(f"Fetching sample data from {gps_start_time} to {gps_end_time} ...")
    if not os.path.isfile(filepath+filename):
        unwhitened_noise = TimeSeries.fetch_open_data(
            ifo,
            gps_start_time - 5,
            gps_end_time + 5,
            sample_rate=srate)

        unwhitened_noise.write(filepath+filename)
    else:
        unwhitened_noise = TimeSeries.read(filepath+filename)
    
    # unwhitened_noise = TimeSeries.fetch_open_data(ifo, gps_start_time, gps_end_time, sample_rate=srate)
    print('duration', unwhitened_noise.duration, len(unwhitened_noise.times), len(unwhitened_noise.value), unwhitened_noise.sample_rate)
    # # Convwerting the unwhitened noise to pycbc for whitening
    whitened_noise = unwhitened_noise.whiten(4, 2)
    whitened_noise = whitened_noise.to_pycbc()

    # plt.plot(whitened_noise[5*srate:-5*srate])

    # # plt.loglog(psd.sample_frequencies, psd)
    # # plt.xlim(10, 4096.)
    # plt.savefig('dummy.png')
    psd = None
    print('OK', len(whitened_noise), whitened_noise.duration)
    # whitened_noise, psd = unwhitened_noise.whiten(
    #     len(unwhitened_noise) / (2 * srate),
    #     len(unwhitened_noise)/( 4 * srate),
    #     remove_corrupted = False,
    #     return_psd = True)


    return unwhitened_noise, whitened_noise, psd

# Example usage

clean_gpstimes = pd.read_csv("../clean_segments/pre_clean_segments_O3a_L1.csv", usecols=['start_time', 'end_time', 'p_values'])


for i in range(len(clean_gpstimes)):
    gps_start_time = clean_gpstimes['start_time'][i]
    gps_end_time = clean_gpstimes['end_time'][i]

    whitened_noise, unwhitened_noise, psd = get_TimeSeries_data(gps_start_time=gps_start_time, gps_end_time=gps_end_time)
    try:
        whitened_noise, unwhitened_noise, psd = get_TimeSeries_data(gps_start_time=gps_start_time, gps_end_time=gps_end_time)
        
        # Print to file
        # with open('output.txt', 'a') as f:
        #     f.write(f"Unwhitened noise: {unwhitened_noise}, Whitened noise: {whitened_noise}, PSD: {psd}\n")
    except Exception as e:
        print(e)
        print(f"Failed to fetch data for {gps_start_time} to {gps_end_time}.")
        break
    print(f"Successfully fetched data for {gps_start_time} to {gps_end_time}.")

    # Print return values
    # print(f"Unwhitened noise: {unwhitened_noise}, Whitened noise: {whitened_noise}, PSD: {psd}")

    