import os
import math as math
import pycbc as pycbc
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
from sklearn import metrics
from typing import Literal
from numpy.typing import NDArray

from ._statistics import get_section_statistics, generate_confusion_matrix
from ._dataset_loader import get_TimeSeries, calculate_q_transform

def display_statistic_pvalue_histogram(pvalues: pd.DataFrame, stat_test: Literal["Shapiro-Wilk", "Kolmogorov-Smirnov"]="Shapiro-Wilk") -> None:
    '''
    A function to plot a histogram of the Shapiro or Kolmogorov-Smirnov p-values from the input.

    Input:
    - `p_values`: A dataframe column containing p-values
    - `statistic`: The statistic being displayed

    Display:
    A histogram of the p-values for the glitch samples 
    '''
    
    pvalues = pvalues.tolist()

    plt.hist(pvalues, bins=40)
    plt.xlabel('Shapiro p-value')
    plt.ylabel('Frequency')
    plt.xticks(list(np.arange(min(pvalues), max(pvalues)+0.01, 0.01)), fontsize=8, rotation=90)
    plt.title(f'Histogram of {stat_test} p-values')
    plt.grid(True)
    plt.show()

    print(f"Number of {stat_test} p-values above 0.05: {(np.array(pvalues) > 0.05).sum()}")
    print(f"Maximum p-value: {max(pvalues)}")
    print(f"Minimum p-value: {min(pvalues)}")

def display_sample_plots(data: pd.DataFrame, save_path: str = "") -> None:
    '''
    A function to display a whitened sample glitch from the input dataframe along its original form and q-transform.

    Input:
    - `data`: A single row of the input dataframe. Must contain the following columns
        - 't': time values for the whitened glitch
        - 'whitened_y': amplitude values of the whitened glitch
        - 'glitch_timeseries': The TimeSeries object for the unwhitened glitch
        - 'q_transform': The q-scan values of the sample

    Display:
    A plot of the whitened glitch, the unwhitened glitch, and the q-transform of the glitch
    '''

    timeseries = TimeSeries.read(data['timeseries_file_location'])
    timeseries = timeseries.whiten(4,2)
    timeseries = timeseries[int(4096 * 4.5):-int(4096 * 4.5)]
    q_scan, time_elapsed = calculate_q_transform(timeseries)

    print(f"Time elapsed for q-transform: {time_elapsed:.2f} seconds")
    
    fig, ax = plt.subplots(3,1, figsize=(12, 24), sharex=True, constrained_layout=True)


    ax[0].plot(data['t'],data['unwhitened_y'])
    ax[0].set_xlabel("Time (s)", fontsize=25)
    ax[0].set_ylabel("Amplitude", fontsize=25)
    ax[0].set_title("Unwhitened Glitch", fontsize=30)
    ax[0].tick_params(axis='both', which="major", labelsize=25)
    ax[0].tick_params(axis='both', which="minor", labelsize=25)
    # ax[0].legend()

    ax[1].plot(data['t'], data['whitened_y'])
    ax[1].set_xlabel("Time (s)", fontsize=25)
    ax[1].set_ylabel("Amplitude", fontsize=25)
    ax[1].set_title("Whitened Glitch", fontsize=30)
    ax[1].tick_params(axis='both', which="major", labelsize=25)
    ax[1].tick_params(axis='both', which="minor", labelsize=25)
    # ax[1].legend()

    ax[2].imshow(q_scan)
    ax[2].set_yscale('log', base=2)
    ax[2].set_xscale('linear')
    ax[2].set_ylabel('Frequency (Hz)', fontsize=25)
    ax[2].set_xlabel('Time (s)', fontsize=25)
    ax[2].images[0].set_clim(0, 25.5)
    ax[2].set_title("Q-Transform", fontsize=30)
    ax[2].tick_params(axis='both', which="major", labelsize=25)
    ax[2].tick_params(axis='both', which="minor", labelsize=25)
    cbr = fig.colorbar(ax[2].images[0], ax=ax[2], label='Normalized energy', orientation='vertical', fraction=0.046, pad=0.05)
    #set colorbar fontsize
    cbr.set_label('Normalized energy', fontsize=25)
    cbr.ax.tick_params(labelsize=25)
    
    if len(save_path):
        plt.savefig(save_path, dpi=100)
    plt.show()


def display_probability_plot(sample: pd.DataFrame) -> None:
    '''
    A function to display the Q-Q/probability plot of a sample glitch from the input dataframe. The input must contain the following information:
    - GPStime
    - y
    - shapiro_pvalue

    Input:
    - `sample`: A **single row** of the input dataframe. Must contain ['GPStime', 'whitened_y', 'shapiro_pvalue'].

    Display:
    A plot of the glitch sample timeseries with sections highlighted to show the  
    '''
    fig,ax = plt.subplots(1,2, figsize=(12,5))
    stats.probplot(sample["whitened_y"], dist="norm", plot=ax[0])
    ax[1].axis("off")
    ax[1].text(0.1, 0.5, f'Shapiro p-value = {sample["shapiro_pvalue"]}\nGPS Time = {sample["GPStime"]}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               fontsize=14, 
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    plt.show()

def display_section_statistics(data: pd.DataFrame, gpstimekey: str = "GPStime", stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro", section_size_seconds: float=1) -> None:
    '''
    A funtion to display one of the following:
    - Shapiro-Wilks Test p-values
    - Kolmogorov-Smirnov Test p-value
    - Anderson-Darling Statistics
    
    of sections of a sample glitch from the input dataframe. The input must contain the following information:
    - t
    - y
    - shapiro_pvalue

    Input:
    - **data:** A **single row** of glitch information. Must contain ['t', 'whitened_y', 'shapiro_pvalue']
    - **stat_test:** The test being performed on the sections (values=["Shapiro", "KS", "Anderson"]). Default="Shapiro".
    - **sections:** The number of sections being studied. Default=1.

    Display:
    A plot of
    - The glitch sample timeseries with sections highlighted to show the concerned statistics for each section.
    - A Q-Q plot of the whole sample
    - Sample Information

    '''

    infotext = ""

    # Calculate section-wise statistics based on section size in seconds
    sectionstats = get_section_statistics(data, stat_test,section_size_seconds)

    fig, ax = plt.subplots(3, 1, figsize=(12,12))
    plt.suptitle(f"{stat_test} Test Statistics for section size={section_size_seconds}")
    ax[0].plot(data['t'], data['whitened_y'])

    for i, section in enumerate(sectionstats):

        if len(section['whitened_y']) > 0:

            if stat_test == "Shapiro" or stat_test == "KS":
                text = f"p={section['section_statistic']['pvalue']:.3f}"
                print(section['section_statistic']['pvalue'])

            elif stat_test == "Anderson":
                if not math.isnan(section['section_statistic']['statistic']):
                    text = f"stat={section['section_statistic']['statistic']:.3f}"
                    print(f"Section {i+1}: \nAD Statistic= {section['section_statistic']['statistic']}\nCritical Values={section['section_statistic']['critical_values']}")

            if not np.isnan(section['section_statistic']['statistic']):
                filled_area = ax[0].fill_between(section['t'], min(section['whitened_y']), max(section['whitened_y']), alpha=0.5)
                (x0, y0), (x1, y1) = filled_area.get_paths()[0].get_extents().get_points()
                ax[0].text((x0 + x1) / 2, y1+8, f'{text}', ha='center', va='center', fontsize=8, color='black')
    
    # Making the info-text for the given data
    if stat_test == "Shapiro":
        infotext = f'GPS Time = {data[gpstimekey]}\nDuration = {data["duration"]}\nKurtosis: {data['kurtosis']}\nSkew: {data['skew']}\nShapiro p-value = {data["shapiro_pvalue"]}'

    elif stat_test == "KS":
        infotext = f'GPS Time = {data[gpstimekey]}\nDuration = {data["duration"]}\nKurtosis: {data['kurtosis']}\nSkew: {data['skew']}\nKS p-value = {data["ks_pvalue"]}'

    elif stat_test == "Anderson":
        ad_stat = stats.anderson(data["whitened_y"], dist='norm')
        infotext = f'GPS Time = {data[gpstimekey]}\nDuration = {data["duration"]}\nKurtosis: {data['kurtosis']}\nSkew: {data['skew']}\nAD Statistic = {data["ad_statistic"]}\nCritical Values={data["ad_critical_values"]}\nSignificance Level={data["ad_significance_level"]}'
    
    stats.probplot(data["whitened_y"], dist="norm", plot=ax[1])

    ax[2].axis('off')
    ax[2].text(0.35, 0.5, infotext, 
            horizontalalignment='left', 
            verticalalignment='center', 
            fontsize=14, 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

def display_confusion_matrix(data: pd.DataFrame, stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro", threshold: float= 0.05,per_glitch:bool=False, save_img: bool=False) -> None:
    '''
    Generate a confusion matrix for the performance of the relevant statistical tests on the signal sample. The statistical tests being considered are
    - Shapiro-Wilks Test
    - Kolmogorov-Smirnov Test
    - Anderson-Darling Test

    Inputs:
    - `data`: The dataset of IFO signal information being studied.

    Display:
    - Confusion matrix for the concerned statistic.
    '''

    if per_glitch:
        title = f"Confusion Matrix of {stat_test} Test on {data.iloc[0]["label"]} Glitches"
        filename = f"conf_matrix_{data.iloc[0]['label']}_{stat_test}.png"
    else:
        title = f"Confusion Matrix of {stat_test} Test"
        filename = f"conf_matrix_{stat_test}.png"
    
    disp = metrics.ConfusionMatrixDisplay(generate_confusion_matrix(data,stat_test, threshold), display_labels=["Glitch Present", "Glitch Not Present"])
    disp.plot()
    plt.grid(False)
    plt.title(title)
    if save_img:
        if not os.path.isdir("./confusion_matrices/"):
            os.mkdir("./confusion_matrices/")
        plt.savefig(f"./confusion_matrices/{filename}")
        plt.close()
    else:
        plt.show()

def display_auc_roc(y_true: pd.DataFrame, data: pd.DataFrame, stat_test: Literal["Shapiro", "KS", "Anderson"]="Shapiro", thresholds: list[float] = []) -> None:
    '''
    Generate the ROC curve for the performance of the relevant statistical tests on the signal samples. The statistical tests being considered are
    - Shapiro-Wilks Test
    - Kolmogorov-Smirnov Test
    - Anderson-Darling Test

    Inputs:
    - `y_true`: The true labels of the dataset.
    - `data`: A datafram containing either p-values for Shapiro-Wilks/Kolmogorov-Smirnov test or Statistic and Critical values for Anderson-Darling test.
    - `stat_test`: The statistic being considered.

    Display:
    - ROC curve for the concerned statistic along with the AUC value.
    '''

    fpr = []
    tpr = []
    if stat_test == "Shapiro":
        title = "Shapiro-Wilk Receiver Operating Characteristic"
    elif stat_test == "KS":
        title = "Kolmogorov-Smirnov Receiver Operating Characteristic"
    elif stat_test == "Anderson":
        title = "Anderson-Darling Receiver Operating Characteristic"

    # Taking defaults of 0.05 for Shapiro and KS test
    # and the 2nd critical value for Anderson-Darling test
    if not len(thresholds):
        if stat_test in ["Shapiro", "KS"]:
            y_pred = data.apply(lambda x: 1 if x <= 0.05 else 0)
        else:
            y_pred = data.apply(lambda x: 1 if x["ad_statistic"] > x["ad_critical_values"][2] else 0, axis=1)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
    # Using input thresholds
    else:
        fpr.append(0)
        tpr.append(0)
        for i in range(len(thresholds)):
            if stat_test in ["Shapiro", "KS"]:
                y_pred = data.apply(lambda x: 1 if x <= thresholds[i] else 0)
            else:
                # For Anderson-Darling, we need to reverse the critical values since it leads to errors
                y_pred = data.apply(lambda x: 1 if x["ad_statistic"] > x["ad_critical_values"][::-1][i] else 0, axis=1)
            
            fpr_temp, tpr_temp, _ = metrics.roc_curve(y_true, y_pred)

            # metrics.roc_curve returns a list of 3 values [0, value, 1]
            # we are only interested in the 2nd value, so we append it
            fpr.append(fpr_temp[1])
            tpr.append(tpr_temp[1])
        
        fpr.append(1)
        tpr.append(1)
        
        print(fpr, tpr)

        # Appending 0 at the beginning and 1 at the end ensures that the
        # Values mimic that from the metrics.roc_curve function for the fpr and tpr
        # This is a bit of a hack but it works :)

        fpr = np.array(fpr).flatten()
        tpr = np.array(tpr).flatten()
    
    roc_auc = metrics.auc(fpr, tpr)

    # plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()