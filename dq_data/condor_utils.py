import numpy as np
import os


def directory_exists(dir_path: str) -> None:
    """Creates directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def create_sh(script_name: str, repo_dir: str, arguments: str, work_dir: str) -> None:
    """
    Create a shell script that will be executed by Condor.
    """
    lines = [
        '#!/bin/bash',
        f'cd {repo_dir}',
        (f'python3 {os.path.join(repo_dir,script_name)} {arguments}')

    ]
    name = script_name.split('/')[1].split('.')[0]
    with open(f"{work_dir}/{name}.sh", 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(f"{work_dir}/{name}.sh", 0o755)
    return name

def create_sub(script_name, sub_arguments, log_directory, accounting_grp, mem, disk, work_dir, repo_dir, name) -> None:
    """
    Create the Condor .sub file.
    """

    lines = [
        'universe              = vanilla',
        'getenv                = true',
        f'request_memory       = {mem}',
        f'request_disk         = {disk}',
        f'executable           = {work_dir}{name}.sh',
        'use_oauth_services    = scitokens',
        (f'environment         = {sub_arguments} BEARER_TOKEN_FILE=$$(CondorScratchDir)/.condor_creds/igwn.use'),
        f'output               = {log_directory}{name[:4]}_$(PID).out', # make the out and error files unique
        f'error                = {log_directory}{name[:4]}_$(PID).err', # following same convention as DAG
        f'log                   = {log_directory}/CreateJobs.log',
        'notification          = never',
        'rank                  = memory',
        f'accounting_group     = {accounting_grp}',
        '',
        'queue 1'
    ]
    #'should_transfer_files = YES',
    #   f'transfer_input_files = {os.path.join(work_dir, "condor",name +".sh" )}, {log_directory}',
    with open(f"{work_dir}{name}.sub", 'w') as f:
        f.write('\n'.join(lines))


def vars_(node_name, dag_lines, kwargs):
    line = "VARS {} ".format(node_name)
    line += " ".join(f'{k}="{v}"' for k, v in kwargs.items())
    dag_lines.append(line)
    dag_lines.append(f"RETRY {node_name} 3")
    dag_lines.append(" ")
    return dag_lines

def python_args(kwargs):
    # '--chunk=${chunk} '

    arguments = " ".join(f'--{k}=${v}' for k, v in kwargs.items())
    return arguments

def sub_args(kwargs):
    # 'chunk=$(chunk);run=$(run);window=$(window);
    arguments = "".join(f'{k}=$({v});' for k, v in kwargs.items())
    return arguments

def job(node_name, dag_lines, script):
    dag_lines.append(f"JOB {node_name} {script}")
    return dag_lines

def create_dag_file(script_name, dag_file, items, kwargs, work_dir, j, name, mode) -> None:
    """
    Create a DAG file to batch multiple jobs.
    For each mode in 'modes', we create a set of jobs for each ID in job_ids.
    """
    dag_lines = []
    if len(items) > 1:
        for count, item in enumerate(items):
            kwargs[list(kwargs.keys())[0]] = count
            if isinstance(item, tuple) is True:
                print(isinstance(item, tuple), 'item')
                for i  in range(len(item)):
                    kwargs[list(kwargs.keys())[j+i]] = item[i]
            else:
                kwargs[list(kwargs.keys())[1]] = item
            node_name = f"{script_name.split('/')[1][:4].upper()}{count}"  # Make each node name unique
            dag_lines = job(node_name, dag_lines, f"{os.path.join(work_dir, name)}.sub")
            dag_lines = vars_(node_name, dag_lines, kwargs)
    else: 
        node_name = f"{script_name.split('/')[1][:4].upper()}"  # Make each node name unique
        dag_lines = job(node_name, dag_lines, f"{os.path.join(work_dir, name)}.sub")
        dag_lines = vars_(node_name, dag_lines, kwargs)
    
    with open(dag_file, mode) as f:
        f.write('\n'.join(dag_lines)+ '\n')

def dependencies(parent_nodes, child_nodes, dag_file, mode):
    dag_lines = f"PARENT {' '.join(parent_nodes)} CHILD {' '.join(child_nodes)}"
    with open(dag_file, mode) as f:
        f.write(dag_lines + '\n') # this is already a string, not a list of strings