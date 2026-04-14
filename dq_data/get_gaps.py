import argparse
import time
import clean_data_utils as utils
import condor_utils as cut
import numpy as np
import os

parser = argparse.ArgumentParser(description="Process gaps for O3a / O3b runs")

parser.add_argument("--run", choices=["O3a", "O3b", "O4a"], required=True, help="Observing run (O3a or O3b or O4a)")
parser.add_argument("--ifo", choices=["L1", "H1", "V1"], default="L1", help="Interferometer (default: L1)")
parser.add_argument("--work_dir", default="/home/melissa.lopez/gwpy-experiments/condor_clean/")
args = parser.parse_args()

cut.directory_exists(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}'))

start, end = utils.get_segment(args.run)
if args.run == 'O4a':
    hoft_channel = f'{args.ifo}:GDS-CALIB_STRAIN_CLEAN'
else:
    hoft_channel = f'{args.ifo}:GDS-CALIB_STRAIN'
lower_bound, upper_bound = 7, 30

stime = time.time()
starts, ends = utils.read_tables(start, end, hoft_channel)
gaps = utils.find_gaps(starts, ends, lower_bound, upper_bound)

print(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}', f'cleangaps_{str(args.run)}_{str(args.ifo)}.npy'))
np.save(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}', f'cleangaps_{str(args.run)}_{str(args.ifo)}.npy'), gaps)
