import math
import pycbc
import numpy as np
import warnings
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter

warnings.filterwarnings('ignore')

__all__ = ["fetch_glitch_data",
           "shapiro_pvalue_histogram",
           "display_glitch_plots",
           "display_subdomain_shapiro",
           "display_probability_plot",
           "subdomain_statistical_study"]

def fetch_glitch_data(data, srate=4096, tw=3, ifo='L1', begin=0, end=50):
    data_readings = []

    # Select information based on begin and end indexes
    g_stars = data['GPStime'].iloc[begin:end]
    durations = data['duration'].iloc[begin:end].to_list()

    for i, g_star in enumerate(g_stars):

        # Fetch noise data from the LIGO GWOSC
        noise = TimeSeries.fetch_open_data(ifo, g_star - tw ,  g_star + tw, sample_rate=srate)
        noise = noise.to_pycbc()

        # whiten the noise data
        white_noise, psd = noise.whiten(len(noise) / (2 * srate),
                                        len(noise)/( 4 * srate),
                                        remove_corrupted = False,
                                        return_psd = True)
        
        # Crop 1s at each side to avoid border effects
        white_noise = white_noise[int(srate * 1):-int(srate * 1)]
        noise = noise[int(srate * 1):-int(srate * 1)]

        # Creating q-transforms of the data for visualization
        data = TimeSeries(white_noise, sample_rate = srate)
        q_scan = data.q_transform(qrange=[4,64], frange=[10, 2048],
                                tres=0.002, fres=0.5, whiten=False)
        
        # Localizing the glitch into a 1s interval
        t = data.times[int(srate * 1.5):-int(srate * 1.5)]
        y = data.value[int(srate * 1.5):-int(srate * 1.5)]
        noise = noise[int(srate * 1.5):-int(srate * 1.5)]

        data_readings.append((g_star,durations[i],noise,y,t,q_scan))

    data_df = pd.DataFrame(data_readings, columns=['GPStime', 'duration', 'noise', 'y', 't', 'q_transform'])
    
    return data_df

def shapiro_pvalue_histogram(data_df, tw):
    plt.hist(data_df['shapiro_pvalue'], bins=40)
    # plt.xscale('log')
    plt.xlabel('Shapiro p-value')
    plt.ylabel('Frequency')
    plt.xticks(list(np.arange(data_df['shapiro_pvalue'].min(), data_df['shapiro_pvalue'].max()+0.01, 0.01)), fontsize=8, rotation=90)
    # plt.xticks(np.arange(0, 0.04*tw, 0.01), fontsize=8)
    # plt.xticks(list(plt.xticks()[0]) + [0.05])
    plt.title('Histogram of Shapiro p-values for Different Glitches')
    plt.grid(True)
    plt.show()
    print(f"Number of Shapiro p-values above 0.05: {(data_df['shapiro_pvalue'] > 0.05).sum()}")
    print(f"Max Shapiro p-value: {data_df['shapiro_pvalue'].max()}")
    print(f"Min Shapiro p-value: {data_df['shapiro_pvalue'].min()}")

def display_glitch_plots(data, glitch_index):
    fig, ax = plt.subplots(1,3, figsize=(24, 6))
    ax[0].plot(data['t'].iloc[glitch_index], data['y'].iloc[glitch_index], label='Suspicious Glitch')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()

    ax[1].plot(data['t'].iloc[glitch_index],data["noise"].iloc[glitch_index], label='Suspicious Glitch')
    ax[2].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend()

    ax[2].imshow(data['q_transform'].iloc[glitch_index])
    ax[2].set_yscale('log', base=2)
    ax[2].set_xscale('linear')
    ax[2].set_ylabel('Frequency (Hz)')
    ax[2].set_xlabel('Time (s)')
    ax[2].images[0].set_clim(0, 25.5)
    fig.colorbar(ax[2].images[0], ax=ax[2], label='Normalized energy', orientation='vertical', fraction=0.046, pad=0.04)

    plt.show()

def display_subdomain_shapiro(data, subdomain_count=9):
    subdomain_size = len(data['t'])//subdomain_count
    subdomain_p_values = []
    list_subdomains = []
    print("Shapiro p-values")
    print("====================")
    print(f"Sample p-value:{data['shapiro_pvalue']}")

    fig, ax = plt.subplots(2,1, figsize=(14,7), dpi=200)
    plt.suptitle(f"Shapiro Test P-values for {subdomain_count} subdomains. Subdomain size={subdomain_size}")
    ax[0].plot(data['t'], data['y'])

    for i in range(0,len(data['y'])+1,subdomain_size):
        x = data['y'][i:i+subdomain_size]
        t = np.array(data['t'])[i:i+subdomain_size]
        list_subdomains.append(x)
        if len(x) > 0:
            subdomain_p_values.append(stats.shapiro(x).pvalue)
            filled_area = ax[0].fill_between(t, min(x), max(x), alpha=0.5)
            (x0, y0), (x1, y1) = filled_area.get_paths()[0].get_extents().get_points()
            if not np.isnan(stats.shapiro(x).pvalue):
                print(stats.shapiro(x).pvalue)
                ax[0].text((x0 + x1) / 2, y1+8, f'p={stats.shapiro(x).pvalue:.3f}', ha='center', va='center', fontsize=8, color='black')
    
    ax[1].axis('off')
    ax[1].text(0.325,
               0.6,
               f'Shapiro p-value = {data["shapiro_pvalue"]}\nGPS Time = {data["GPStime"]}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               fontsize=14, 
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    
    plt.show()
    
    return list_subdomains

def display_probability_plot(sample_glitch):
    fig,ax = plt.subplots(1,2, figsize=(12,5))
    stats.probplot(sample_glitch["y"], dist="norm", plot=ax[0])
    ax[1].axis("off")
    ax[1].text(0.1, 0.5, f'Shapiro p-value = {sample_glitch["shapiro_pvalue"]:.10f}\nGPS Time = {sample_glitch["GPStime"]}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               fontsize=14, 
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    plt.show()

def subdomain_statistical_study(data, stat_test="Shapiro", subdomains=1):
    '''
    A function to calculate the  takes in:
    - **data:** A row of glitch information
    - **stat_test:** The test being performed on the subdomains (values=["Shapiro", "KS", "Anderson"], default="Shapiro")
    - **subdomains:** The number of subdomains being studied (default=1)
    
    Response
      - Test results related to each of the subdomains of the dataset
    '''
    
    subdomain_size = len(data['y'])//subdomains
    subdomain_statistics = []
    subdomain_statistic = {}
    infotext=""
    skew_kurtosis = f"Kurtosis: {stats.kurtosis(data['y'], fisher=False)}\nSkew: {stats.skew(data['y'])}"

    print(f"{stat_test} Statistics")
    print("====================")
    # print(stats.ks_1samp(data["y"],stats.norm.cdf))
    fig, ax = plt.subplots(3, 1, figsize=(12,12))
    plt.suptitle(f"{stat_test} Test Statistics for {subdomains} subdomains. Subdomain size={subdomain_size}")
    ax[0].plot(data['t'], data['y'])

    for count,i in enumerate(range(0, len(data['y']+1), subdomain_size)):
        y = data['y'][i:i+subdomain_size]
        t = np.array(data['t'])[i:i+subdomain_size]
        if len(y) > 0:
            if stat_test == "Shapiro":
                subdomain_statistic = stats.shapiro(y)
                text = f"p={subdomain_statistic.pvalue:.3f}"
                print(subdomain_statistic.pvalue)
            elif stat_test == "KS":
                subdomain_statistic = stats.ks_1samp(y,stats.norm.cdf)
                text = f"p={subdomain_statistic.pvalue:.3f}"
                print(subdomain_statistic.pvalue)
            elif stat_test == "Anderson":
                subdomain_statistic = stats.anderson(y, dist='norm')
                if not math.isnan(subdomain_statistic.statistic):
                    text = f"p={subdomain_statistic.statistic:.3f}"
                    print(f"Subdomain {count+1}: \nAD Statistic= {subdomain_statistic.statistic}\nCritical Values={subdomain_statistic.critical_values}")

            subdomain_statistics.append(subdomain_statistic)
            
            if not np.isnan(subdomain_statistic.statistic):
                filled_area = ax[0].fill_between(t, min(y), max(y), alpha=0.5)
                (x0, y0), (x1, y1) = filled_area.get_paths()[0].get_extents().get_points()
                ax[0].text((x0 + x1) / 2, y1+8, f'{text}', ha='center', va='center', fontsize=8, color='black')
    
    stats.probplot(data["y"], dist="norm", plot=ax[1])

    if stat_test == "Shapiro":
        infotext = f'GPS Time = {data["GPStime"]}\nDuration = {data["duration"]}\nShapiro p-value = {data["shapiro_pvalue"]}\n'
    elif stat_test == "KS":
        infotext = f'GPS Time = {data["GPStime"]}\nDuration = {data["duration"]}\nKS p-value = {stats.ks_1samp(data["y"],stats.norm.cdf).pvalue}\n'
    elif stat_test == "Anderson":
        ad_stat = stats.anderson(data["y"], dist='norm')
        infotext = f'GPS Time = {data["GPStime"]}\nDuration = {data["duration"]}\nAD Statistic = {ad_stat.statistic}\nCritical Values={ad_stat.critical_values}\nSignificance Level={ad_stat.significance_level}\n'

    ax[2].axis('off')
    ax[2].text(0.35, 0.5, infotext+skew_kurtosis, 
            horizontalalignment='left', 
            verticalalignment='center', 
            fontsize=14, 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    return subdomain_statistics
