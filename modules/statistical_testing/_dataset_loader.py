import os
import math as math
import pycbc as pycbc
import numpy as np
import warnings as warnings
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from gwpy.table import GravitySpyTable
from numpy.typing import NDArray

from ._statistics import calculate_sample_statistics

warnings.filterwarnings('ignore')

def get_TimeSeries(gpstime: float, tw: int=5, srate=4096, ifo='L1') -> list:
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

    filepath = "./glitch_timeseries_data/"
    filename = f"sample_{ifo}_{gpstime}_{tw}.h5"
    q_scan = {}

    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    
    # Check if TimeSeries information is already loaded.
    # Fetch noise data from the LIGO GWOSC if not present
    if not os.path.isfile(filepath+filename):
        print(f"Fetching data for GPS time {gpstime} ...")
        unwhitened_noise = TimeSeries.fetch_open_data(ifo, gpstime - tw,  gpstime + tw, sample_rate=srate)
        unwhitened_noise.write(filepath+filename)
    else:
        unwhitened_noise = TimeSeries.read(filepath+filename)

    unwhitened_noise = unwhitened_noise.to_pycbc()

    # whiten the noise data
    whitened_noise, psd = unwhitened_noise.whiten(
        len(unwhitened_noise) / (2 * srate),
        len(unwhitened_noise)/( 4 * srate),
        remove_corrupted = False,
        return_psd = True)
    
    # Crop times at each side to avoid border effects
    whitened_noise = whitened_noise[int(srate * (tw - 2)):-int(srate * (tw - 2))]
    unwhitened_noise = unwhitened_noise[int(srate * (tw - 2)):-int(srate * (tw - 2))]

    # Creating q-transforms of the data for visualization
    whitened_noise = TimeSeries(whitened_noise, sample_rate = srate)
    
    # Try block added because some glitches throw:
    # ValueError "Input signal contains non-numerical values"
    try:
        q_scan = whitened_noise.q_transform(qrange=[4,64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=False)
    except ValueError:
        print(f"Failed to generate q-transform for {gpstime}")

    # Stripping the sample down to a 1 second window of central data
    # (this might need to be changed for different glitches)
    unwhitened_noise = unwhitened_noise[int(srate * 1.5):-int(srate * 1.5)]
    whitened_noise = whitened_noise[int(srate * 1.5):-int(srate * 1.5)]

    return unwhitened_noise, whitened_noise, q_scan, psd
        

def fetch_glitch_data_from_csv(data: pd.DataFrame, gpsTimeKey: str="GPStime", tw: int=5, srate=4096, ifo='L1', begin=0, n_samples=0)-> pd.DataFrame:

    '''
    Fetches the glitch TimeSeries samples from the TimeSeries API, performs the statistical tests on them retruns a datset with the relevant information appended 

    Inputs:
    - `data`: Pandas dataframe of glitch data
    - `gpsTimeKey`: The key value for GPS time in the dataset
    - `begin`: Starting index. Default=0
    - `n_samples`: The number of samples to be taken. Default=0 (loads all the data)

    Output: The original input dataset concatenated with the following
    - 'y': Amplitude values of the whitened glitch timeseries
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
            unwhitened_noise, whitened_noise, q_scan, psd = get_TimeSeries(g_star, tw=tw, srate=srate, ifo=ifo)
            t = whitened_noise.times
            y = whitened_noise.value

            # Fetching relevant data to be appended to the input dataframe
            supplemental_glitch_data = {
                "glitch_timeseries": unwhitened_noise,
                "y": y,
                "t": t,
                "q_scan": q_scan}
        
            supplemental_glitch_data.update(calculate_sample_statistics(whitened_noise))
        except ValueError:
            supplemental_glitch_data = {
                "glitch_timeseries": np.nan,
                "y": np.nan,
                "t": np.nan,
                "q_scan": np.nan,
                "shapiro_statistic": np.nan,
                "shapiro_pvalue": np.nan,
                "shapiro_prediction": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "ks_prediction": np.nan,
                "ad_statistic": np.nan,
                "ad_critical_values": np.nan,
                "ad_significance_level": np.nan,
                "kurtosis": np.nan,
                "skew": np.nan
            }
            print(f"Failed to load data for {g_star}. Glitch Type = {data_copy.iloc[i]['label']}")

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

def fetch_clean_segment_samples(data ,ifo:str="L1", sample_rate: int=4096, segment_duration_seconds: float = 1) -> pd.DataFrame:
    ''''
    Fetches Timeseries segments for all rows of the input DataFrame and returns a dataframe of whitened samples from it at a given segment size.
    '''

    filepath = "./clean_timeseries_data/"

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    segment_size = sample_rate*segment_duration_seconds
    fail_count = 0

    whitened_samples = []
    print("Input Length: ",len(data))
    
    for i in range(len(data)):
        if data.iloc[i]['start_time'] and data.iloc[i]['end_time']:
            
            filename = f"clean_sample_{ifo}_{data.iloc[i]['start_time']}_{data.iloc[i]['end_time']}.h5"

            # Fetching the whole GWOSC timeseries of the given segment
            try:
                if not os.path.isfile(filepath+filename):
                    unwhitened_sample = TimeSeries.fetch_open_data(
                        ifo,
                        data.iloc[i]['start_time'],
                        data.iloc[i]['end_time'],
                        sample_rate=sample_rate)
                    unwhitened_sample.write(filepath+filename)
                else:
                    unwhitened_sample = TimeSeries.read(filepath+filename)
                unwhitened_sample = unwhitened_sample.to_pycbc()
            except ValueError as e:
                print(e)
                continue
        
            # Whitening the sample segment
            try:
                whitened_sample, psd = unwhitened_sample.whiten(
                    len(unwhitened_sample) / (2 * sample_rate),
                    len(unwhitened_sample) / (4 * sample_rate),
                    remove_corrupted=False,
                    return_psd=True
                )
            except ValueError as e:
                # print(f"Failed to whiten sample for {data.iloc[i]['start_time']}-{data.iloc[i]['end_time']}")
                data.iloc[i]["load_failed"] = 1
                fail_count = fail_count + 1
                print(e)
                continue

            whitened_sample = TimeSeries(whitened_sample, sample_rate = sample_rate)

            # Getting segments of the whitened sample equal to the input segment duration
            # This will serve as the clean segments for the our statistical tests
            if not len(whitened_sample) < segment_size:
                for i in range(0, len(whitened_sample.times) + 1, segment_size):
                    
                    # Only accept samples that are of the exact segment size
                    if not i < segment_size:
                        whitened_samples.append(whitened_sample[i:i + segment_size])
        
    print("Number of failed samples: ", fail_count)
    print("Number of whitened samples obtained: ", len(whitened_samples))
            
    whitened_samples_df = pd.DataFrame(whitened_samples, columns=['whitened_sample_timeseries'])
    return whitened_samples_df
