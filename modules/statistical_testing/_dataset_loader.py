import os
import math as math
import pycbc as pycbc
import numpy as np
import warnings as warnings
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from gwpy.table import GravitySpyTable
from numpy.typing import NDArray

from ._statistics import calculate_sample_statistics

warnings.filterwarnings('ignore')

def get_TimeSeries(gps_time: float, gps_end_time: float=0, tw: int=5, srate=4096, ifo='L1', bandpass: bool=False) -> list:
    '''
    This function fetches data from the GWOSC TimeSeries API and stores them in "./glitch_timeseries_data" corresponding to the sample if not already present.

    Inputs:
    - `gpstime`: The GPS time of the sample
    - `tw`: Time window to be taken into consideration on either side of the glitch. Thefinal sample returned will have 2.5 seconds removed from either side.
            Default = 5 seconds 
    - `srate`: The sampling rate. Default= 4096
    - `ifo`: The interferometer being studied. Default=L1 (LIGO LivingstonObservatory)

    Outputs:
    - `unwhitened_noise`: TimeSeries object of the sample
    - `whitened_noise`: : TimeSeries object of the whitened sample
    - `q_scan`: q-scan of the sample
    - `psd`: The calculated power spectral density of the sample

    '''

    filepath = "./timeseries_data/"
    q_scan = {}
    
    # Create the directory if it does not exist
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    
    # Check if TimeSeries information is already loaded.
    # Fetch noise data from the LIGO GWOSC if not present
    if not gps_end_time:
        filename = f"sample_{ifo}_{gps_time}_{tw}.h5"
        start_time = gps_time - tw
        end_time = gps_time + tw
        # print(f"Fetching sample data for GPS time {gps_time} with a {tw} second time window ...")
    else:
        start_time = gps_time # End time already provided
        end_time = gps_end_time
        filename = f"sample_{ifo}_{gps_time}_{end_time}.h5"
        # print(f"Fetching sample data from {gps_time} to {end_time} ...")

    if not os.path.isfile(filepath+filename):
        unwhitened_noise = TimeSeries.fetch_open_data(
            ifo,
            start_time,
            end_time,
            sample_rate=srate)
        unwhitened_noise.write(filepath+filename)
    else:
        unwhitened_noise = TimeSeries.read(filepath+filename)

    # whitening the noise data
    # try:
    #     whitened_noise, psd = unwhitened_noise.whiten(
    #         len(unwhitened_noise) / (2 * srate),
    #         len(unwhitened_noise)/( 4 * srate),
    #         remove_corrupted = False,
    #         return_psd = True)
    # except ValueError as e:
    #     # In case of failuer retry with changed TimeSeries length
    #     print(f"Failed to whiten sample for {gps_time}.")
    #     print(e)
    #     return [], [], {}, 0

    # # Converting the unwhitened noise to pycbc for whitening
    # unwhitened_noise = unwhitened_noise.to_pycbc()
    
    # Whitening the noise data
    whitened_noise = unwhitened_noise.whiten(4, 2)

    if bandpass:
        # Band pass between 50 to 250 Hz for Scattered Light glitches
        bp = filter_design.bandpass(15, 250, srate)
        whitened_noise = whitened_noise.filter(bp)

    # Old method of whitening
    # # Computing the PSD with Welch's method. I start at 8Hz
    # psd = unwhitened_noise.filter_psd(unwhitened_noise.duration, unwhitened_noise.delta_f, flow=8)
    # # According to me this smoothes the PSD to be able to whiten
    # psd = pycbc.psd.estimate.inverse_spectrum_truncation(psd,
    #                max_filter_len= int(1*srate),
    #                trunc_method='hann', low_frequency_cutoff=None,)
    # # We whiten the data
    # whitened_noise = (unwhitened_noise.to_frequencyseries() / psd ** 0.5).to_timeseries()

    
    # Cropping the data to remove border effects
    # If end_time is not provided, we crop it down to 2 seconds on either side for q_scan calculations
    if not gps_end_time:
        # Crop times at each side down to 4 seconds for q_scans
        whitened_noise = whitened_noise[int(srate * (tw - 2)):-int(srate * (tw - 2))]
        unwhitened_noise = unwhitened_noise[int(srate * (tw - 2)):-int(srate * (tw - 2))]
    else:
        # Crop 1 second at each side to avoid border effects
        whitened_noise = whitened_noise[int(srate * 1):-int(srate * 1)]
        unwhitened_noise = unwhitened_noise[int(srate * 1):-int(srate * 1)]
    
    # Conversion to GWPY for q-transform calculation
    whitened_noise = TimeSeries(whitened_noise, sample_rate = srate)
    unwhitened_noise = TimeSeries(unwhitened_noise, sample_rate = srate)

    # Creating q-transforms of the data for visualization
    # This is not done if the end time is provided
    # Try block added because some glitches throw ValueErrors
    if not gps_end_time:
        try:
            q_scan = calculate_q_transform(whitened_noise)
        except ValueError as e:
            print(f"Failed to generate q-transform for {gps_time}")
            print(e)

        # Stripping the sample down to a 1 second window of central data
        # (this might need to be changed for different glitches)
        unwhitened_noise = unwhitened_noise[int(srate * 1.5):-int(srate * 1.5)]
        whitened_noise = whitened_noise[int(srate * 1.5):-int(srate * 1.5)]

    return unwhitened_noise, whitened_noise, q_scan

def calculate_q_transform(sample: TimeSeries):
    '''
    This function fetches data from the GWOSC TimeSeries API and returns the q-transform of the sample

    Inputs:
    - `sample`: The whitened timeseries sample

    Outputs:
    - `q_scan`: q-scan of the sample
    '''
    q_scan = sample.q_transform(qrange=[4,64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=False)
    return q_scan


def fetch_glitch_data_from_csv(data: pd.DataFrame, gpsTimeKey: str="GPStime", tw: int=5, srate=4096, ifo='L1', begin=0, n_samples=0, bandpass: bool=False)-> pd.DataFrame:

    '''
    Fetches the glitch TimeSeries samples from the TimeSeries API, performs the statistical tests on them retruns a datset with the relevant information appended 

    Inputs:
    - `data`: Pandas dataframe of glitch data
    - `gpsTimeKey`: The key value for GPS time in the dataset
    - `begin`: Starting index. Default=0
    - `n_samples`: The number of samples to be taken. Default=0 (loads all the data)

    Output: The original input dataset concatenated with the following
    - 'unwhitened_y': Amplitude values of the glitch timeseries
    - 'whitened_y': Amplitude values of the whitened glitch timeseries
    - 't': Time values of the whitened glitch timeseries,
    - 'q_transform': Q-transform of the whole glitch sample (1 second removed at either end to account for border effects)
    - 'Shapiro-Wilk statistic': Shapiro statistic of the sample amplitudes
    - 'Shapiro-Wilk p-value': Shapiro p-value of the sample amplitudes
    - 'Shapiro-Wilk prediction': The prediction made based on the Shapiro-Wilks p-value of sample amplitudes
    - 'Kolmogorov-Smirnov Statistic': Kolmogorov-Smirnov statistic of the sample amplitudes
    - 'Kolmogorov-Smirnov p-value': Kolmogorov-Smirnov p-value of the sample amplitudes
    - 'Kolmogorov-Smirnov prediction': The prediction made based on the Kolmogorov-Smirnov p-value of sample amplitudes
    - 'Anderson-Darling statistic': Anderson-Darling statistic
    - 'Anderson-Darling critical values': Critical values for the Anderson Darling statistic
    - 'Anderson-Darling significance level': Significance level for the Anderson Darling statistic
    - 'Anderson-Darling prediction': The prediction made based on the Anderson-Darling statistic
    - 'Kurtosis': Kurtosis of the glitch amplitude values
    - 'Skew': Skew of the glitch amplitude values
    '''

    # Create a copy of the original dataset, selecting the relevant information to work on
    data_copy = data[data['ifo'].str.casefold() == ifo.casefold()].copy(deep=True).reset_index(drop=True)

    data_readings = {}

    # Select the rows based on beginning index and number of samples needed
    # g_stars = data_copy[gpsTimeKey].iloc[begin:n_samples]

    i = 0

    if not n_samples:
        limit = len(data_copy)
    else:
        limit = min(n_samples, len(data_copy))

    while i < limit:
        unwhitened_noise = []
        whitened_noise = []
        q_scan = []
        psd = 0
        
        g_star = data_copy.iloc[i][gpsTimeKey]

        # Try clause is the normal program flow
        # Except clause skips the current iteration and enters zero values
        # if TimeSeries fails to load
        try:
            # Applying the bandpass filter to the Scattered Light data if needed
            # if bandpass and data_copy.iloc[i]['label'] == 'Scattered_Light':
            #     unwhitened_noise, whitened_noise, q_scan = get_TimeSeries(g_star, tw=tw, srate=srate, ifo=ifo, bandpass=True)
            # else:
            #     unwhitened_noise, whitened_noise, q_scan = get_TimeSeries(g_star, tw=tw, srate=srate, ifo=ifo)

            unwhitened_noise, whitened_noise, q_scan = get_TimeSeries(g_star, tw=tw, srate=srate, ifo=ifo, bandpass=bandpass)
            
            t = whitened_noise.times
            whitened_y = whitened_noise.value
            unwhitened_y = unwhitened_noise.value

            # Fetching relevant data to be appended to the input dataframe
            supplemental_glitch_data = {
                "unwhitened_y": unwhitened_y,
                "whitened_y": whitened_y,
                "t": t,
                "q_scan": q_scan}
        
            supplemental_glitch_data.update(calculate_sample_statistics(whitened_y))
        except ValueError as e:
            supplemental_glitch_data = {
                "unwhitened_y": np.nan,
                "whitened_y": np.nan,
                "t": np.nan,
                "q_scan": np.nan,
                "shapiro_statistic": np.nan,
                "shapiro_pvalue": np.nan,
                # "shapiro_prediction": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                # "ks_prediction": np.nan,
                "ad_statistic": np.nan,
                "ad_critical_values": np.nan,
                "ad_significance_level": np.nan,
                # "ad_prediction": np.nan,
                "kurtosis": np.nan,
                "skew": np.nan
            }
            print(f"Failed to load data for {g_star}. Glitch Type = {data_copy.iloc[i]['label']}")
            print(e)

        for key, value in supplemental_glitch_data.items():
            if key in data_readings:
                data_readings[key].append(value)
            else:
                data_readings[key] = [value]

        i = i + 1
        
    # Append generated statistics to current dataframe and return it        
    dataframe_to_append = pd.DataFrame.from_dict(data_readings, orient='columns').reset_index(drop=True)
    if n_samples:
        data_df = pd.concat([data.iloc[begin:n_samples].reset_index(drop=True), dataframe_to_append], axis=1)
    else:
        data_df = pd.concat([data.reset_index(drop=True), dataframe_to_append], axis=1)

    data_df["glitch_present"] = 1

    return data_df

def fetch_gspy_glitch_data(glitchtype: str) -> None:
    filepath = f"./gspy_glitches/gspy_{glitchtype}.csv"
    load_dotenv(find_dotenv())

    GRAVITYSPY_DATABASE_USER = os.getenv('GRAVITYSPY_DATABASE_USER')
    GRAVITYSPY_DATABASE_PASSWD = os.getenv('GRAVITYSPY_DATABASE_PASSWD')
    
    if not os.path.exists("./gspy_glitches"):
        os.makedirs("./gspy_glitches")
    if not os.path.exists(filepath):
        glitch_data = GravitySpyTable.fetch(
            "gravityspy",
            "glitches",
            selection=f"ml_label={glitchtype}"
        ).to_pandas()
        glitch_data.to_csv(filepath, index=False)

def fetch_clean_segment_samples(data ,ifo:str="L1", sample_rate: int=4096, segment_duration_seconds: float=1, n_samples: int=50, bandpass: bool=False) -> pd.DataFrame:
    ''''
    Fetches Timeseries segments for all rows of the input DataFrame and returns a dataframe of whitened samples from it at a given segment size.
    '''

    filepath = "./timeseries_data/"

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    segment_size = sample_rate * segment_duration_seconds
    fail_count = 0

    whitened_samples = []
    print("Input Length: ",len(data))
    
    for i in range(len(data)):
        
        try:
            unwhitened_sample, whitened_sample, q_scan = get_TimeSeries(data.iloc[i]['start_time'], gps_end_time=data.iloc[i]['end_time'], srate=sample_rate, ifo=ifo, bandpass=bandpass)
        except ValueError as e:
            print(f"Failed to load data for segment {data.iloc[i]['start_time']} - {data.iloc[i]['end_time']}.")
            print(e)
            fail_count += 1
            continue

        # Skip in the case of errors loading whitened data
        if not len(whitened_sample):
            continue

        # Getting segments of the whitened sample equal to the input segment duration
        # This will serve as the clean segments for the our statistical tests
        if not len(whitened_sample) < segment_size:
            for i in range(0, len(whitened_sample.times) + 1, segment_size):
    
                # Only accept samples that are of the exact segment size
                if not i < segment_size: #FIXME
                    sample = whitened_sample[i:i + segment_size]

                    segment_data = {
                        "y": sample.value,
                        "t": sample.times,
                    }

                    segment_data.update(calculate_sample_statistics(segment_data['y']))
                    whitened_samples.append(segment_data)
        

    # whitened_samples_df = pd.DataFrame(columns=['unwhitened_sample_timeseries', 'unwhitened_sample_timeseries'])

    # Randomly select n_samples from the generated whitened samples
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(whitened_samples), size=n_samples, replace=False)
    whitened_samples_df = pd.DataFrame([whitened_samples[i] for i in selected_indices])

    print("Number of failed samples: ", fail_count)
    print("Number of whitened samples obtained: ", len(whitened_samples_df))

    whitened_samples_df["glitch_present"] = 0

    return whitened_samples_df
