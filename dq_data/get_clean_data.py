from gwtrigfind import find_trigger_files
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
from scipy.optimize import curve_fit
import re
from dq_utils import get_DQ_segments
import argparse
import h5py


def read_tables(start, end, hoft_channel):
    """
    Reads only the GPS start and end times from multiple HDF5 trigger files.
    Returns a single sorted numpy array of starts and ends.
    """
    file_list = find_trigger_files(hoft_channel, 'omicron', start, end)
    starts_list, ends_list = [], []

    for file in file_list:
        with h5py.File(file[7:], "r") as f:
            triggers = f["triggers"]
            starts_list.append(triggers["tstart"][:])
            ends_list.append(triggers["tend"][:])

    # Concatenate all files
    starts, ends = np.concatenate(starts_list), np.concatenate(ends_list)

    # Sort by start times
    sort_idx = np.argsort(all_starts)
    starts, ends = starts[sort_idx], ends[sort_idx]

    return starts, ends

def find_gaps(starts, ends, lower_bound, upper_bound):
    """
    Vectorized gap calculation between consecutive triggers.
    
    Parameters:
        starts (np.ndarray): start times of triggers
        ends (np.ndarray): end times of triggers
        lower_bound (float): min gap in seconds
        upper_bound (float): max gap in seconds

    Returns:
        np.ndarray: shape (N,2) of valid gaps [gap_start, gap_end]
    """
    # Compute gaps vectorized
    gap_starts = ends[:-1]
    gap_ends   = starts[1:]
    gaps       = gap_ends - gap_starts

    # Apply bounds
    valid_mask = (gaps > lower_bound) & (gaps < upper_bound)

    # Return array of (gap_start, gap_end)
    return np.column_stack((gap_starts[valid_mask], gap_ends[valid_mask]))


def get_o3_segment(run):
    """
    Return (start, end) GPS times for O3a or O3b.
    """

    if run == "O3a":
        start = 1238166018  # O3a start
        end   = 1253977218  # O3a end
    elif run == "O3b":
        start = 1256655618  # O3b start
        end   = 1269363618  # O3b end
    else:
        raise ValueError("run must be 'O3a' or 'O3b'")

    return start, end


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
def main(run, ifo):
    """
    Main execution function that orchestrates gap processing.
    """
    start, end = get_o3_segment(run)
    hoft_channel = f'{ifo}:GDS-CALIB_STRAIN'
    lower_bound, upper_bound = 7, 30

    starts, ends = read_tables(start, end, hoft_channel)
    gaps = find_gaps(starts, ends, lower_bound, upper_bound)
    p_values = process_gaps(gaps, scratch=2, plot=False)
    print(len(df), len(gaps))
    # data = pd.DataFrame(gaps, columns=['start_time', 'end_time'])
    # data['p_values'] = p_values
    # data.to_csv(f'pre_clean_segments_O3a_{ifo}.csv')

    # Create a pandas data frame
    gaps = pd.DataFrame(gaps, columns=['start_time', 'end_time'])
    gaps['p_values'] = p_values

    # Data quality (DQ) segments are also returned in pandas
    segments = get_DQ_segments(ifo, start, end)

    # Check if each row in df2 falls within any interval in df1
    mask = gaps.apply(lambda row: ((segments['start_time'] <= row['start_time']) & (segments['end_time'] >= row['end_time'])).any(), axis=1)

    # Filter df2 to get only matching rows
    DQ_gaps = gaps[mask]

    DQ_gaps.to_csv(f'pre_clean_segments_{run}_{ifo}_new.csv')

# Run the script if executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gaps for O3a / O3b runs")

    parser.add_argument("--run", choices=["O3a", "O3b"], required=True, help="Observing run (O3a or O3b)")
    parser.add_argument("--ifo", choices=["L1", "H1", "V1"], default="L1", help="Interferometer (default: L1)")
    args = parser.parse_args()

    main(run=args.run, ifo=args.ifo)
