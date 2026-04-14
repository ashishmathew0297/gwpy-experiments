import argparse
import pandas as pd
import time
import numpy as np
import os
from dq_utils import get_DQ_segments
import clean_data_utils as utils
import condor_utils as cut

parser = argparse.ArgumentParser(description="Process gaps for O3a / O3b runs")

parser.add_argument("--run", choices=["O3a", "O3b", "O4a"], required=True, help="Observing run (O3a or O3b or O4a)")
parser.add_argument("--ifo", choices=["L1", "H1", "V1"], default="L1", help="Interferometer (default: L1)")
parser.add_argument("--job", required=True, help="Job ID")
parser.add_argument("--num_gaps", default=1000, help="Number of gaps to process in a single job")
parser.add_argument("--work_dir", default="/home/melissa.lopez/gwpy-experiments/condor_clean/")

args = parser.parse_args()

cut.directory_exists(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}'))

job, num_gaps = int(args.job), int(args.num_gaps)
gaps = np.load(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}',f'cleangaps_{args.run}_{args.ifo}.npy'))
gaps_ = gaps[num_gaps * job : num_gaps * (job + 1)]
gaps_ = pd.DataFrame(gaps_, columns=['start_time', 'end_time'])

# Data quality (DQ) segments are also returned in pandas
start, end = utils.get_segment(args.run)
segments = get_DQ_segments(args.ifo, start, end, args.run)

# Check if each row in df2 falls within any interval in df1
mask = gaps_.apply(lambda row: ((segments['start_time'] <= row['start_time']) & (segments['end_time'] >= row['end_time'])).any(), axis=1)

# Filter df2 to get only matching rows
gaps_dq = gaps_[mask]
gaps_np = np.array([gaps_dq['start_time'], gaps_dq['end_time']])
print(gaps_np.shape, args.num_gaps)
p_values = utils.process_gaps(args.ifo, args.run, gaps_np,
                              scratch=2, plot=False)
#print(gaps_dq['start_time'].values, gaps_dq['end_time'].values)
print(p_values)
gaps_dq['p_values'] = p_values

gaps_dq.to_csv(os.path.join(args.work_dir, f'{str(args.run)}_{str(args.ifo)}', f'pre_clean_segments_{args.run}_{args.ifo}_{args.job}_new.csv'))
