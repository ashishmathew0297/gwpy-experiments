### Instructions

In this folder we acquire clean and glitch data following CAT2 data quality (see [my thesis](https://research-portal.uu.nl/ws/portalfiles/portal/245781352/phddissertationmelissalopez%20-%206739e41d4dc0a.pdf) p. 33 and references therein.).

#### Clean data

To get the clean data first we need to get the segments where there is no Omicron triggers. Afterwards, we compute the p-value for each segment using `glitchfind` code. The workflow to get clean data is `get_gaps.py --> statistic_computation.py --> merge_gaps.py`. The final product is a `csv` file containing all clean segments with their respetive p-values. Note that if a p-value <= 0.05 the data is not considered clean and cannot be used.

- How to run: With `workflow.uni` and `make_dag.py` we create a DAG to process `get_gaps.py --> statistic_computation.py`. Relevant arguments are `run` and `ifo`. Note that you can only get Omicron information from the actual `ifo` and not CIT. Once your DAG is created, before submission we need to run `htgettoken -a vault.ligo.org -i igwn` to get an scitoken. This token only works for 3h, so otherwise we need computing help if the job is too long.

- Misc: Info about calibration in O4 is [here](https://wiki.ligo.org/LSC/JRPComm/ObsRun4#A_42Calibrated_strain_channels_42)