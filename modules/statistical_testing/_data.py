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

warnings.filterwarnings('ignore')

def fetch_glitch_data_from_csv(data: pd.DataFrame, gpsTimeKey: str="GPStime", tw: int=5, srate=4096, ifo='L1', begin=0, end=50):

    '''
    Fetches the glitch TimeSeries samples from the TimeSeries API, performs the statistical tests on them retruns a datset with the relevant information appended 

    Inputs:
    - `data`: Pandas dataframe of glitch data
    - `gpsTimeKey`: The key value for GPS time in the dataset
    - `srate`: The sampling rate. Default= 4096
    - `tw`: Time window to be taken into consideration on either side of the glitch. Thefinal sample returned will have 2.5 seconds removed from either side.
            Default = 5 seconds 
    - `ifo`: The interferometer being studied. Default=L1 (LIGO LivingstonObservatory)
    - `begin`: Starting index. Default=0
    - `end`: End index. Default=50

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

    data_readings = {}
    columns_to_append = ["glitch_timeseries","y","t","q_scan","shapiro_statistic", "shapiro_pvalue","shapiro_prediction","ks_statistic","ks_pvalue","ks_prediction","ad_statistic","ad_critical_values","ad_significance_level","kurtosis","skew"]

    # Select information based on begin and end indexes
    g_stars = data[gpsTimeKey].iloc[begin:end]

    for i, g_star in enumerate(g_stars):
        # Fetch noise data from the LIGO GWOSC
        unwhitened_noise = TimeSeries.fetch_open_data(ifo, g_star - tw,  g_star + tw, sample_rate=srate)
        unwhitened_noise = unwhitened_noise.to_pycbc()

        # whiten the noise data
        whitened_noise, psd = unwhitened_noise.whiten(len(unwhitened_noise) / (2 * srate),
                                        len(unwhitened_noise)/( 4 * srate),
                                        remove_corrupted = False,
                                        return_psd = True)
        
        # Crop times at each side to avoid border effects
        whitened_noise = whitened_noise[int(srate * 3):-int(srate * 3)]
        unwhitened_noise = unwhitened_noise[int(srate * 3):-int(srate * 3)]

        # Creating q-transforms of the data for visualization
        whitened_noise = TimeSeries(whitened_noise, sample_rate = srate)

        # Try block added because some glitches throw:
        # ValueError "Input signal contains non-numerical values"
        try:
            q_scan = whitened_noise.q_transform(qrange=[4,64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=False)
        except ValueError:
            print(f"Failed to load data for {g_star}")
        
        # Stripping the sample down to 1 second of central data
        # (this might need to be changed for different glitches)
        unwhitened_noise = unwhitened_noise[int(srate * 1.5):-int(srate * 1.5)]
        whitened_noise = whitened_noise[int(srate * 1.5):-int(srate * 1.5)]
        
        t = whitened_noise.times
        y = whitened_noise.value

        # Fetching relevant data to be appended to the input dataframe
        supplemental_glitch_data = {
            "glitch_timeseries": unwhitened_noise,
            "y": y,
            "t": t,
            "q_scan": q_scan}
        
        supplemental_glitch_data.update(generate_sample_statistics(whitened_noise))

        for key, value in supplemental_glitch_data.items():
            if key in data_readings:
                 data_readings[key].append(value)
            else:
                data_readings[key] = [value]

    # Append generated statistics to current dataframe and return it        
    dataframe_to_append = pd.DataFrame.from_dict(data_readings, orient='columns').reset_index(drop=True)
    data_df = pd.concat([data.iloc[begin:end].reset_index(drop=True), dataframe_to_append], axis=1)
    
    return data_df

def fetch_gspy_glitch_data(glitchtype: str):
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

def generate_sample_statistics(sample_timeseries: TimeSeries) -> dict:
    '''
    This function uses the input glitch TimeSeries sample in pycbc form to calculate and return a list of the following

    - Shapiro-Wilks statistic
    - Shapiro-Wilks p-value
    - Kolmogorov-Smirnov statistic
    - Kolmogorov-Smirnov p-value
    - Anderson-Darling statistic
    - Anderson-Darling critical values
    - Anderson-Darling significance levels

    Inputs:
    - `glitch_timeseries`: TimeSeries object of the glitch

    Output:
    - **data_df**: a list containing the following
        - 'shapiro_statistic': Shapiro-Wilks statistic of the sample amplitudes
        - 'shapiro_pvalue': Shapiro-Wilks p-value of the sample amplitudes
        - 'ks_pvalue': Kolmogorov-Smirnov p-value of the sample amplitudes
        - 'ad_statistic': Anderson-Darling statistic
        - 'ad_critical_values': Critical values for the Anderson Darling statistic
        - 'ad_significance_level': Significance level for the Anderson Darling statistic
        - 'kurtosis': Kurtosis of the glitch amplitude values
        - 'skew': Skew of the glitch amplitude values
    '''

    np.random.seed(42)
    
    y = sample_timeseries.value

    # =================== Shapiro-Wilks Test ===================

    sw_statistic = stats.shapiro(y)

    # =================== Two-Sample Kolmogorov-Smirnov Test ===================

    # The Kolmogorov Smirnov statistic needs to be applied to a scaled
    # version of our data to work properly since it is a distance-based
    # metric
    scaler = MinMaxScaler(feature_range=(-4,4))
    ks_statistic = stats.ks_2samp(list(scaler.fit_transform(y.reshape(-1,1))[:,0]), stats.norm.rvs(size=len(y), random_state=np.random.default_rng()))

    # =================== Anderson-Darling Test ===================

    ad_statistic = stats.anderson(y, dist='norm')

    # =================== Skew and Kurtosis ===================

    kurtosis = stats.kurtosis(y, fisher=False)
    skew = stats.skew(y)

    return {
        "shapiro_statistic": sw_statistic.statistic,
        "shapiro_pvalue": sw_statistic.pvalue,
        "shapiro_prediction": 1 if sw_statistic.pvalue <= 0.05 else 0,
        "ks_statistic": ks_statistic.statistic,
        "ks_pvalue": ks_statistic.pvalue,
        "ks_prediction": 1 if ks_statistic.pvalue <= 0.05 else 0,
        "ad_statistic": ad_statistic.statistic,
        "ad_critical_values": ad_statistic.critical_values,
        "ad_significance_level": ad_statistic.significance_level,
        "kurtosis": kurtosis,
        "skew": skew
    }

def get_section_statistics(data: pd.DataFrame, stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro", section_size_seconds: float=1) -> list:
    '''
    A function to calculate one of the following:
    - Shapiro-Wilks Test p-values
    - Kolmogorov-Smirnov Test p-value
    - Anderson-Darling Statistics
    
    for sections of a sample glitch from the input dataframe generated by `fetch_glitch_data()`

    Input:
    - **data:** A **single row** of glitch information. Must contain ['t', 'y']
    - **stat_test:** The test being performed on the sections (values=["Shapiro", "KS", "Anderson"]). Default="Shapiro".
    - **section_size_seconds:** The number of sections (in seconds) being studied. The accepted values range from 0 (exclusive) to 1 with a maximum precision of 4. Default=1 second.

    Display: A plot of
    - The glitch sample timeseries with sections highlighted to show the concerned statistics for each section.
    - A Q-Q plot of the whole sample

    Output:
      - **section_statistics:** A list of test results in relation to each of the sections of the dataset.
    '''

    section_info = []
    section_statistic = {}
    sample_length = len(data['y'])

    # Section size (in seconds) rounded to 5 places
    section_size_seconds = round(section_size_seconds, 5)
    
    # Using the sample timeframe in seconds, get section size
    # in terms of sampling rate
    if section_size_seconds <= sample_length/4096 and section_size_seconds > 0:
        section_size = int(math.floor(sample_length) * section_size_seconds)
    else:
        section_size = sample_length

    print(f"{stat_test} Statistics")
    print("====================")

    # =================== Section-wise Statistics Calculation ===================

    for i in range(0, len(data['y']+1), section_size):

        y = data['y'][i:i+section_size]
        t = np.array(data['t'])[i:i+section_size]

        if len(y) > 0:
            if stat_test == "Shapiro":
                section_statistic = stats.shapiro(y)._asdict()
                text = f"p={section_statistic['pvalue']:.3f}"
            elif stat_test == "KS":
                section_statistic = stats.ks_1samp(y,stats.norm.cdf)._asdict()
                text = f"p={section_statistic['pvalue']:.3f}"
            elif stat_test == "Anderson":
                section_statistic = stats.anderson(y, dist='norm')._asdict()
            
            if not np.isnan(section_statistic['statistic']):
                section_info.append({"y":y, "t":t, "section_statistic":section_statistic})

    return section_info

def get_confusion_matrix(data: pd.DataFrame, stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro") -> NDArray:
    '''
    Generate a confusion matrix for the performance of the relevant statistical tests on the signal sample. The statistical tests being considered are
    - Shapiro-Wilks Test
    - Kolmogorov-Smirnov Test
    - Anderson-Darling Test

    Inputs:
    - `data`: The dataset of IFO signal information being studied.
    - `stat_test`: The statistical test being considered.

    Output:
    - Confusion matrix for the concerned statistic.
    '''

    cm = []

    if stat_test == "Shapiro":
        cm = metrics.confusion_matrix(np.ones(len(data)),data["shapiro_prediction"],labels=[1,0])
    if stat_test == "KS":
        cm = metrics.confusion_matrix(np.ones(len(data)),data["ks_prediction"],labels=[1,0])
    # TODO: Decide on significance level for AD statistic
    
    return cm