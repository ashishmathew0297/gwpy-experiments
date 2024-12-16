import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from tqdm import tqdm
from gwpy.timeseries import TimeSeries
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Paths to datasets with Gravisty Spy confidences > 0.9
DATA_PATHS = {
    "O3a": "gspy_glitch_datasets/data_o3a_high_confidence.csv",
    "O3b": "gspy_glitch_datasets/data_o3b_high_confidence.csv",
}

# Constants
NUM_SAMPLES = 3 # Let's plot 3 examples per class
SAMPLE_RATE = 4096
IFOS = ["H1", "L1"]
SEGMENT_DURATION = 8 # stretch of data in seconds to consider
CROP_DURATION = 2 # How many seconds to remove from each end of the whitened strain, due to corruption from filter
ANALYSIS_DURATION = 1 # duration of data to consider for Shapiro test, Shapiro starts to break with N > 5000, so 1s (N=4096) is almost maximum that we can do with current sampling rate
PLOTS_PATH = "shapiro_plots/"
CROP_LENGTH = CROP_DURATION * SAMPLE_RATE 
ANALYSIS_LENGTH = ANALYSIS_DURATION * SAMPLE_RATE
PLOT_EXAMPLES = True

# Load datasets
datasets = {name: pd.read_csv(path).drop_duplicates() for name, path in DATA_PATHS.items()}
glitch_labels = list(datasets["O3a"].label.unique())

# Initialize nested dictionary to store Shapiro p-values
shapiro_p_values = {dataset_name: {ifo: {} for ifo in IFOS} for dataset_name in DATA_PATHS.keys()}

def fetch_glitch_data(ifo, gps_time, sample_rate, segment_duration):
    """
    Fetch glitch and ASD data from the specified IFO and GPS time.
    """
    glitch = TimeSeries.fetch_open_data(
        ifo,
        gps_time - segment_duration // 2, # 4s either side of the trigger time
        gps_time + segment_duration // 2,
        sample_rate=sample_rate
    )

    # We calculate the ASD for whitening using 8s just before the analysis segment, if we include the glitch in the ASD calculation, it will be surpressed during whitening. 
    segment_for_asd = TimeSeries.fetch_open_data(
        ifo,
        gps_time - 3*segment_duration // 2,
        gps_time - segment_duration // 2,
        sample_rate=sample_rate
    )
    return glitch, segment_for_asd.asd()

def process_glitch_data(glitch, asd=None):
    """
    Whiten, crop, and extract the analysis portion of the glitch data with a given asd.
    If no asd is given, the asd will be calculated with the glitch included (NOT RECOMMENDED)
    """
    if asd==None:
        asd = glitch.asd()
    white_glitch = glitch.whiten(asd=asd)
    white_glitch = np.array(white_glitch)
    white_glitch_crop = white_glitch[CROP_LENGTH:-CROP_LENGTH] # Remove the corrupted to make sure, just in case analysis durations are changed later
    middle_of_glitch = len(white_glitch_crop) // 2
    white_glitch_for_analysis = white_glitch_crop[
        middle_of_glitch - ANALYSIS_LENGTH // 2 : middle_of_glitch + ANALYSIS_LENGTH // 2
    ]
    return white_glitch_for_analysis - np.mean(white_glitch_for_analysis) # Let's make sure we have 0 mean

def save_plot(glitch_data, dataset_name, ifo, glitch_type, index, p_value):
    """
    Save a plot of the glitch data with Shapiro p-value annotated.
    """
    plotting_dir = os.path.join(PLOTS_PATH, dataset_name, ifo, glitch_type)
    os.makedirs(plotting_dir, exist_ok=True)

    plt.figure()
    plt.plot(glitch_data)
    plt.title(f'{glitch_type}_{index}, Shapiro p-value = {p_value:.3f}')
    plt.savefig(os.path.join(plotting_dir, f'glitch_type_{index}.png'))
    plt.close()

# Main loop, let's cycle through observing runs, ifos and glitch types
for dataset_name, dataset in datasets.items():
    for ifo in IFOS:
        data_filtered = dataset[dataset.ifo == ifo]

        for glitch_type in glitch_labels:
            # Print progress dynamically
            print(f'\rProcessing: Dataset={dataset_name}, IFO={ifo}, Glitch Type={glitch_type}{" " * 20}', end='')

            # Filter for the current glitch type
            data_glitch = data_filtered[data_filtered.label == glitch_type]
            if data_glitch.empty:
                continue

            data_glitch.reset_index(drop=True, inplace=True)
            random_indices = np.random.choice(len(data_glitch), size=min(NUM_SAMPLES, len(data_glitch)), replace=False)

            for idx in random_indices:
                gps_time = data_glitch.GPStime.iloc[idx]

                try:
                    # Fetch glitch and ASD data
                    glitch, asd = fetch_glitch_data(ifo, gps_time, SAMPLE_RATE, SEGMENT_DURATION)
                except ValueError as e:
                    print(f"\nSkipping GPS time {gps_time} for IFO {ifo} due to error: {e}")
                    continue

                # Process glitch data, whitening and centering around 1s
                centered_glitch = process_glitch_data(glitch, asd)

                # Perform Shapiro test
                _, shapiro_p_value = shapiro(centered_glitch)
                shapiro_p_values[dataset_name][ifo].setdefault(glitch_type, []).append(shapiro_p_value)

                # Save plots if required
                if PLOT_EXAMPLES:
                    save_plot(centered_glitch, dataset_name, ifo, glitch_type, idx, shapiro_p_value)

# Save Shapiro p-values to a JSON file
output_file = "shapiro_p_values.json"
with open(output_file, "w") as f:
    json.dump(shapiro_p_values, f, indent=4)

print(f"\nShapiro p-values saved to {output_file}")
