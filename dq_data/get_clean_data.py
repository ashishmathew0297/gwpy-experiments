from gwtrigfind import find_trigger_files
from gwpy.table import (Table, EventTable)
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
from scipy.optimize import curve_fit
import re
from dq_utils import get_DQ_segments


def read_table(start, end, hoft_channel):
    trigger_files = find_trigger_files(hoft_channel, 'omicron', start, end)
    trigger_table = EventTable.read(trigger_files,
                                     tablename='sngl_burst',
                                     format='ligolw')
    df = trigger_table.to_pandas()
    df['peak_gps'] = df["peak_time"] + df["peak_time_ns"]*10**(-9)
    df['start_gps'] = df["start_time"] + df["start_time_ns"]*10**(-9)
    df['end_gps'] = df['start_gps'] + df['duration']
    return df

def find_gaps(df, lower_bound, upper_bound):
    """
    Find gaps in a dataframe where the difference between 'end_gps' of one row 
    and 'start_gps' of the next row falls within the given bounds.
    
    Parameters:
    df (pd.DataFrame): A dataframe with 'start_gps' and 'end_gps' columns.
    lower_bound (float): Minimum gap size.
    upper_bound (float): Maximum gap size.

    Returns:
    list: A list of tuples containing (gap_start, gap_end).
    """
    # Compute the differences using vectorized operations
    gap_starts = df["end_gps"].iloc[:-1].to_numpy()
    gap_ends = df["start_gps"].iloc[1:].to_numpy()
    gaps = gap_ends - gap_starts

    # Filter gaps based on the given bounds
    valid_mask = (gaps > lower_bound) & (gaps < upper_bound)
    
    # Extract valid (gap_start, gap_end) tuples
    return list(zip(gap_starts[valid_mask], gap_ends[valid_mask]))

def _monoLog(x, m, t):
    """
    Monotonic logarithmic model.
    """
    return np.log(m) - (t * x)

def num_exp(x, L, m, t):
    """
    Exponential function used to compute a rate.
    """
    return L * m / t * np.exp(-t * x)

def p_val(x, L, params):
    """
    Calculate the p-value based on the exponential rate.
    """
    rate = num_exp(x, L, *params)
    return 1 - np.exp(-rate)

def fit_qgram_data(qgram, fit_bins=15, min_bin=2, max_bin=42, p0=(3, 0.5)):
    """
    Fit the data from qgram and calculate the p-value.

    Parameters:
    qgram (dict): Data containing the energy values.
    fit_bins (int): Number of bins for the fit.
    min_bin (int): Minimum bin value.
    max_bin (int): Maximum bin value.
    p0 (tuple): Initial guess for the parameters m and t.

    Returns:
    float: p-value for the fitted parameters.
    """
    # Prepare the histogram data (without plotting)
    n, _ = np.histogram(qgram["energy"], bins=100, density=True)
    
    # Create bins and bin midpoints
    bin_num = int(max_bin - min_bin) * 2 + 1
    bins = np.linspace(min_bin, max_bin, bin_num)
    bin_mids = (bins[:-1] + bins[1:]) / 2.0

    # Prepare the y-values for fitting
    fit_yvals = np.ma.log(n[: int(fit_bins)])
    fit_yvals = fit_yvals.filled(fit_yvals.min() - 1)

    # Fit the curve
    params, _ = curve_fit(_monoLog, bin_mids[: int(fit_bins)], fit_yvals, p0=p0)
    m, t = params

    # Calculate the p-value
    max_energy = np.max(qgram["energy"])
    p_value = p_val(max_energy, len(qgram), (m, t))

    # Return the p-value in string format
    if p_value < 0.000001:
        return "< 0.000001"
    else:
        return "%.6f" % p_value
    
def process_gaps(gaps, scratch=2, plot=True):
    """
    Process gaps by fetching and whitening time series data, computing q-grams, 
    calculating p-values, and optionally plotting results.

    Parameters:
    gaps (list of tuples): List of (start, end) GPS time gaps.
    scratch (int, optional): Extra time buffer before and after each gap. Default is 2 seconds.
    plot (bool, optional): Whether to plot the results. Default is True.

    Returns:
    list: List of calculated p-values as strings.
    """
    cmap = colormaps['viridis']
    p_values = []

    for i in range(len(gaps)):
        start = gaps[i][0] - scratch
        end = gaps[i][1] + scratch
        print(f"Processing gap {i+1}/{len(gaps)}: Duration = {end - start} seconds")

        # Fetch and whiten data
        try:
            data = TimeSeries.fetch_open_data('L1', start, end)
            data = data.whiten(4, 2)

            # Define parameters
            q_low, q_high = 4, 64
            f_low, f_high = 10, 2048

            # Compute q-gram
            qgram = data.q_gram(qrange=(q_low, q_high),
                                snrthresh=2,
                                frange=(f_low, f_high),
                                mismatch=0.5)

            # Fit qgram data and compute p-value
            p_value_str = fit_qgram_data(qgram)
            p_value_str = float(re.search(r"\d+\.\d+", p_value_str).group())
        except ValueError as e:
            print(f"ValueError encountered: {e}")
            p_value_str = 0
            
        p_values.append(p_value_str)

        if plot:
            # Generate the q-gram plot
            plot = qgram.tile('time', 'frequency', 'duration', 'bandwidth',
                              color='energy', label=f"Calculated p-value: {p_value_str}")
            ax = plot.gca()

            # Format axes
            ax.set_xscale('seconds')
            ax.set_yscale('log')
            ax.set_ylim(16, 1024)
            ax.set_ylabel('Frequency [Hz]')
            ax.grid(True, axis='y', which='both')
            ax.set_facecolor(cmap(0))  # Color background to the bottom of the colormap

            # Colorbar
            ax.colorbar()

            # Mark gap start/end
            plt.axvline(gaps[i][0], color='crimson', label="Gap Start")
            plt.axvline(gaps[i][1], color='crimson', label="Gap End")

            # Finalize plot
            plt.legend()
            plot.show()

    return p_values  # Returns the list of p-values


# Main function
def main():
    """
    Main execution function that orchestrates gap processing.
    """

    start = 1238166018 #O3a start
    end = 1253977218 #O3a end
    ifo = "L1"
    hoft_channel = f'{ifo}:GDS-CALIB_STRAIN'
    lower_bound, upper_bound = 7, 30

    df = read_table(start, end, hoft_channel)
    gaps = find_gaps(df, lower_bound, upper_bound)
    p_values = process_gaps(gaps, scratch=2, plot=False)
    
    data = pd.DataFrame(gaps, columns=['start_time', 'end_time'])
    data['p_values'] = p_values
    data.to_csv(f'pre_clean_segments_O3a_{ifo}.csv')
    
    start = 1238166018 #O3a start
    end = 1253977218 #O3a end
    ifo = "L1"
    hoft_channel = f'{ifo}:GDS-CALIB_STRAIN'
    lower_bound, upper_bound = 7, 30

    df = read_table(start, end, hoft_channel)
    gaps = find_gaps(df, lower_bound, upper_bound)
    p_values = process_gaps(gaps, scratch=2, plot=False)

    # Create a pandas data frame
    gaps = pd.DataFrame(gaps, columns=['start_time', 'end_time'])
    gaps['p_values'] = p_values

    # Data quality (DQ) segments are also returned in pandas
    segments = get_DQ_segments(ifo, start, end)

    # Check if each row in df2 falls within any interval in df1
    mask = gaps.apply(lambda row: ((segments['start_time'] <= row['start_time']) & (segments['end_time'] >= row['end_time'])).any(), axis=1)

    # Filter df2 to get only matching rows
    DQ_gaps = gaps[mask]

    DQ_gaps.to_csv(f'pre_clean_segments_O3a_{ifo}.csv')

# Run the script if executed directly
if __name__ == "__main__":
    main()