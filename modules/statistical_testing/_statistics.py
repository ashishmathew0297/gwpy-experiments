import os
import math as math
import pycbc as pycbc
import numpy as np
import warnings as warnings
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from typing import Literal
from dotenv import find_dotenv, load_dotenv
from gwpy.table import GravitySpyTable
from numpy.typing import NDArray

warnings.filterwarnings('ignore')

def calculate_sample_statistics(y_values: list) -> dict:
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
    # scaler = MinMaxScaler(feature_range=(-4,4))
    scaler = StandardScaler()

    scaled_y_values = list(scaler.fit_transform(np.array(y_values).reshape(-1,1))[:,0])

    # =================== Shapiro-Wilks Test ===================

    sw_statistic = stats.shapiro(y_values)
    scaled_sw_statistic = stats.shapiro(scaled_y_values)

    # =================== Two-Sample Kolmogorov-Smirnov Test ===================

    # The Kolmogorov Smirnov statistic needs to be applied to a scaled
    # version of our data to work properly since it is a distance-based
    # metric
    ks_statistic = stats.ks_2samp(scaled_y_values, stats.norm.rvs(size=len(y_values)))

    # =================== Anderson-Darling Test ===================

    ad_statistic = stats.anderson(y_values, dist='norm')

    # =================== Skew and Kurtosis ===================

    kurtosis = stats.kurtosis(y_values, fisher=False)
    skew = stats.skew(y_values)

    return {
        "shapiro_statistic": sw_statistic.statistic,
        "shapiro_pvalue": sw_statistic.pvalue,
        "shapiro_prediction": 1 if sw_statistic.pvalue <= 0.05 else 0,
        "scaled_shapiro_statistic": scaled_sw_statistic.statistic,
        "scaled_shapiro_pvalue": scaled_sw_statistic.pvalue,
        "scaled_shapiro_prediction": 1 if scaled_sw_statistic.pvalue <= 0.05 else 0,
        "ks_statistic": ks_statistic.statistic,
        "ks_pvalue": ks_statistic.pvalue,
        "ks_prediction": 1 if ks_statistic.pvalue <= 0.05 else 0,
        "ad_statistic": ad_statistic.statistic,
        "ad_critical_values": ad_statistic.critical_values,
        "ad_significance_level": ad_statistic.significance_level,
        "kurtosis": kurtosis,
        "skew": skew
    }

def get_section_statistics(data: pd.DataFrame, stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro", section_duration_seconds: float=1) -> list:
    '''
    A function to calculate one of the following:
    - Shapiro-Wilks Test p-values
    - Kolmogorov-Smirnov Test p-value
    - Anderson-Darling Statistics
    
    for sections of a sample glitch.

    Input:
    - **data:** A **single row** of glitch information. Must contain ['t', 'whitened_y']
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
    sample_length = len(data['whitened_y'])

    # Section size (in seconds) rounded to 5 places
    section_duration_seconds = round(section_duration_seconds, 5)
    
    # Using the sample timeframe in seconds, get section size
    # in terms of sampling rate
    # Checks:
    # 1. If section duration is less than or equal to the sample length in seconds
    # 2. If section duration is greater than 0
    # If both conditions are satisfied, calculate the section size
    # else use the whole sample as the section
    if section_duration_seconds <= sample_length/4096 and section_duration_seconds > 0:
        section_size = int(math.floor(sample_length) * section_duration_seconds)
    else:
        section_size = sample_length

    print(f"{stat_test} Statistics")
    print("====================")

    # =================== Section-wise Statistics Calculation ===================

    for i in range(0, len(data['whitened_y']+1), section_size):

        y = data['whitened_y'][i:i+section_size]
        t = np.array(data['t'])[i:i+section_size]

        # Calculating the section statistics
        if len(y) > 0:
            if stat_test == "Shapiro":
                section_statistic = stats.shapiro(y)._asdict()
            elif stat_test == "KS":
                # scaler = MinMaxScaler(feature_range=(-4,4))
                scaler = StandardScaler()
                section_statistic = stats.ks_2samp(list(scaler.fit_transform(y.reshape(-1,1))[:,0]), stats.norm.rvs(size=len(y)))._asdict()
            elif stat_test == "Anderson":
                section_statistic = stats.anderson(y, dist='norm')._asdict()

            if not np.isnan(section_statistic['statistic']):
                section_info.append({"whitened_y":y, "t":t, "section_statistic":section_statistic})

    return section_info

def generate_confusion_matrix(data: pd.DataFrame, stat_test: Literal["Shapiro", "Shapiro_scaled", "KS", "Anderson"]="Shapiro") -> NDArray:
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
    if stat_test == "Shapiro_scaled":
        cm = metrics.confusion_matrix(np.ones(len(data)),data["shapiro_prediction"],labels=[1,0])
    if stat_test == "KS":
        cm = metrics.confusion_matrix(np.ones(len(data)),data["ks_prediction"],labels=[1,0])
    # TODO: Decide on significance level for AD statistic
    
    return cm

def generate_evaluation_metrics(confusion_matrix):
    """
    Prints evaluation metrics given a confusion matrix.
    
    Parameters:
    confusion_matrix (list of list of int or scipy sparse matrix): 2x2 confusion matrix
    """
    # Convert scipy sparse matrix to dense if necessary
    if issparse(confusion_matrix):
        confusion_matrix = confusion_matrix.toarray()
    
    # Extract values from the confusion matrix
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Print metrics
    return accuracy, precision, recall, f1_score