import configparser
import os
from pathlib import Path
import condor_utils as cut
import sys
import math
import numpy as np

cfg = configparser.ConfigParser()
cfg.read("workflow.ini")

# -----------------------
# GLOBALS
# -----------------------
python = cfg["GLOBAL"]["python"]
work_dir = cfg["GLOBAL"]["work_dir"]
repo_dir = cfg["GLOBAL"]["repo_dir"]
run = cfg["GLOBAL"]["run"].replace("'","")
ifo = cfg["GLOBAL"]["ifo"].replace("'","")

condor_dir = cfg["CONDOR"]["condor_dir"]
log_dir = cfg["CONDOR"]["log_dir"]

cut.directory_exists(condor_dir)
cut.directory_exists(log_dir)
    
accounting_group = cfg["CONDOR"]["accounting_group"]
mem = cfg.getint("CONDOR", "request_memory")
disk = cfg.getint("CONDOR", "request_disk")

dag_lines, vars_lines = [], []
dag_file = f"{work_dir}/main.dag"

# # -----------------------
# # TRAINING SUBDAG
# # -----------------------
gaps = cfg['GAPS']
kwargs = {"work_dir": "work_dir", "run": "run", "ifo": "ifo"} # to generate sh and sub files
kwargs_dag = {"work_dir": work_dir, "run": run, "ifo":ifo} # to generate dag
args_sh, args_sub = cut.python_args(kwargs), cut.sub_args(kwargs)

items_gaps = [1] # there is just one job
name = cut.create_sh(gaps['script'], repo_dir=repo_dir, arguments=args_sh, work_dir=work_dir)
cut.create_sub(gaps['script'], args_sub, log_dir, accounting_group, mem, disk, work_dir, repo_dir, name)
cut.create_dag_file(gaps['script'], dag_file, items_gaps, kwargs_dag, work_dir, j=1, name=name, mode='w') # appending lines

# # -----------------------
# # STATS SUBDAG
# # -----------------------

np_file_name = os.path.join(work_dir, f'{run}_{ifo}', f'cleangaps_{run}_{ifo}.npy')
np_file = np.load(np_file_name)

stats = cfg['STATS']
num_gaps = int(stats['num_gaps'])
jobs = math.ceil(len(np_file)/num_gaps)

kwargs = {"work_dir": "work_dir", "run": "run", "ifo": "ifo", "job":"job", "num_gaps": "num_gaps"} # to generate sh and sub files
kwargs_dag = {"PID":None, "job":None, "work_dir": work_dir, "run": run, "ifo": ifo, "num_gaps": num_gaps} # to generate dag
args_sh, args_sub = cut.python_args(kwargs), cut.sub_args(kwargs)

items_stats = np.arange(jobs)
print(items_stats)
name = cut.create_sh(stats['script'], repo_dir=repo_dir, arguments=args_sh, work_dir=work_dir)
cut.create_sub(stats['script'], args_sub, log_dir, accounting_group, mem, disk, work_dir, repo_dir, name)
cut.create_dag_file(stats['script'], dag_file, items_stats, kwargs_dag, work_dir, j=1, name=name, mode='a') # appending lines

# # -----------------------
# # Dependencies
# # -----------------------

parent_nodes = [gaps['script'].split('/')[1][:4].upper()]
child_nodes = [f"{stats['script'].split('/')[1][:4].upper()}{i}" for i in range(len(items_stats))]

cut.dependencies(parent_nodes, child_nodes, dag_file, 'a') # appending lines

# All done
print('condor_submit_dag', dag_file)

