import numpy as np
import os


def directory_exists(dir_path: str) -> None:
    """Creates directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def create_sh(script_name: str, repo_dir: str, arguments: str, work_dir: str) -> None:
    """
    Create a shell (.sh) script that will later be executed by HTCondor.

    The script:
    1. Changes to the repository directory.
    2. Executes the target Python script with provided arguments.

    Parameters
    ----------
    script_name : str
        Path to the python script (relative to repo_dir).
    repo_dir : str
        Path to the repository directory.
    arguments : str
        Command-line arguments passed to the python script.
    work_dir : str
        Directory where the generated .sh file will be stored.

    Returns
    -------
    name : str
        Base name of the script (used for .sub file naming).
    """
    lines = [
        '#!/bin/bash',
        f'cd {repo_dir}',
        (f'python3 {os.path.join(repo_dir,script_name)} {arguments}')

    ]
    # Extract base script name (without path or extension)
    name = script_name.split('/')[1].split('.')[0]
    with open(f"{work_dir}/{name}.sh", 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(f"{work_dir}/{name}.sh", 0o755) # Make the script executable
    return name

def create_sub(script_name, sub_arguments, log_directory, accounting_grp, mem, disk, work_dir, repo_dir, name) -> None:
    """
    Create an HTCondor submission (.sub) file.

    This file defines how a single job should be executed on the cluster.

    Parameters
    ----------
    script_name : (str) Name of the original Python script.

    sub_arguments : (str) Environment variable definitions passed via Condor.

    log_directory : (str) Directory where: stdout (.out), stderr (.err), Condor log (.log)files will be written.

    accounting_grp : (str) Condor accounting group. Required on shared clusters for quota and usage tracking.

    mem : (str or int) Requested memory for the job.

    disk : (str or int) Requested disk space for the job.
       
    work_dir : (str) Directory where: the executable .sh file lives and the .sub file will be written

    repo_dir : (str) Repository directory.

    name : (str) Base name of the job.
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
    """
    Add variable definitions and retry policy for a DAG node.

    Parameters
    ----------
    node_name : str
        Name of the DAG node.
    dag_lines : list
        Existing DAG file lines.
    kwargs : dict
        Variables passed into the node.

    Returns
    -------
    Updated dag_lines list.
    """
    line = "VARS {} ".format(node_name)
    line += " ".join(f'{k}="{v}"' for k, v in kwargs.items())
    dag_lines.append(line)
    dag_lines.append(f"RETRY {node_name} 3")  # Allow up to 3 retries on failure
    dag_lines.append(" ")
    return dag_lines

def python_args(kwargs):
    """
    Convert keyword dictionary into Python CLI arguments
    for Condor macro substitution.

    Example:
        {'chunk': 'chunk'} → '--chunk=${chunk}'
    """

    arguments = " ".join(f'--{k}=${v}' for k, v in kwargs.items())
    return arguments

def sub_args(kwargs):
    """
    Convert keyword dictionary into Condor environment assignments.

    Example:
        {'chunk': 'chunk'} → 'chunk=$(chunk);'
    """
    arguments = "".join(f'{k}=$({v});' for k, v in kwargs.items())
    return arguments

def job(node_name, dag_lines, script):
    """
    Add a JOB entry to the DAG file.

    Parameters
    ----------
    node_name : str
        Name of the DAG node.
    script : str
        Path to the .sub file.
    """
    dag_lines.append(f"JOB {node_name} {script}")
    return dag_lines

def create_dag_file(script_name, dag_file, items, kwargs, work_dir, j, name, mode) -> None:
    """
    Create (or append to) a DAG file.

    If multiple items are provided, create one DAG node per item.
    Supports tuple expansion for multiple parameter updates.

    Parameters
    ----------
    script_name : str
        Python script name.
    dag_file : str
        Path to DAG file.
    items : list
        List of job parameter values.
    kwargs : dict
        Parameter dictionary for variable substitution.
    work_dir : str
        Directory where .sub file lives.
    j : int
        Offset index used when unpacking tuple items.
    name : str
        Base job name.
    mode : str
        File mode ('w' for write, 'a' for append).
    """
    dag_lines = []
    if len(items) > 1:
        for count, item in enumerate(items):
            # First kwarg key gets loop index
            kwargs[list(kwargs.keys())[0]] = count
            
            # If item is a tuple, unpack into multiple kwargs
            if isinstance(item, tuple) is True:
                print(isinstance(item, tuple), 'item')
                for i  in range(len(item)):
                    kwargs[list(kwargs.keys())[j+i]] = item[i]
            else:
                # Otherwise assign second key
                kwargs[list(kwargs.keys())[1]] = item
            # Unique node name (first 4 letters of script, uppercase)
            node_name = f"{script_name.split('/')[1][:4].upper()}{count}"  # Make each node name unique
            dag_lines = job(node_name, dag_lines, f"{os.path.join(work_dir, name)}.sub")
            dag_lines = vars_(node_name, dag_lines, kwargs)
    else:
        # Unique node name (first 4 letters of script, uppercase)
        node_name = f"{script_name.split('/')[1][:4].upper()}"  # Make each node name unique
        dag_lines = job(node_name, dag_lines, f"{os.path.join(work_dir, name)}.sub")
        dag_lines = vars_(node_name, dag_lines, kwargs)
    
    with open(dag_file, mode) as f:
        f.write('\n'.join(dag_lines)+ '\n')

def dependencies(parent_nodes, child_nodes, dag_file, mode):
    """
    Define DAG dependencies.

    Creates:
        PARENT parent1 parent2 CHILD child1 child2

    Meaning:
        Children run only after parents finish successfully.
    """
    dag_lines = f"PARENT {' '.join(parent_nodes)} CHILD {' '.join(child_nodes)}"
    with open(dag_file, mode) as f:
        f.write(dag_lines + '\n') # this is already a string, not a list of strings