import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Process gaps for O3a / O3b runs")

parser.add_argument("--run", choices=["O3a", "O3b", "O4a"], required=True, help="Observing run (O3a or O3b or O4a)")
parser.add_argument("--ifo", choices=["L1", "H1", "V1"], default="L1", help="Interferometer (default: L1)")
parser.add_argument("--work_dir", default="/home/melissa.lopez/gwpy-experiments/condor_clean/")
parser.add_argument("--repo_dir", default="/home/melissa.lopez/gwpy-experiments/dq_data/")
args = parser.parse_args()

files_dir = os.path.join(args.work_dir, f'{args.run}_{args.ifo}')

Df = pd.DataFrame()
for file in os.listdir(files_dir):
    if 'pre_clean' in file:
        df = pd.read_csv(os.path.join(files_dir, file), index_col=0)
        Df = pd.concat([Df, df])
print(len(Df))
Df = Df.sort_values(by='start_time').reset_index(drop=True).drop_duplicates(subset=['start_time', 'end_time'])
print(len(Df))
Df.to_csv(os.path.join(args.repo_dir, f'pre_clean_segments_{args.run}_{args.ifo}_new.csv'))
