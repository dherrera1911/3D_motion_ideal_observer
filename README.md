## Optimal motion-in-depth estimation with natural stimuli

This is the code associated with the paper
["Optimal motion-in-depth estimation with natural stimuli"](https://www.biorxiv.org/content/10.1101/2024.03.14.585059v1.abstract),
by [Daniel Herrera-Esposito](https://dherrera1911.github.io/) and
[Johannes Burge](https://jburge.psych.upenn.edu/); *bioRxiv*, 2023

The code is organized into three main directories according to
three different tasks:
* `code/stimulus_synthesis/`: Generate naturalistic
binocular videos used in the experiments
* `code/training_scripts/`: Training ideal observer model (AMA).
There's one script for each task
* `code/plotting_scripts/`: Evaluating the trained AMA models and
generate the paper plots

## Getting the data

Stimuli datasets (synthesized with code in `stimulus_synthesis`)
can be downloaded from the companion [OSF repository](https://osf.io/w9mpe/).
The datasets should be placed in directory `data/ama_inputs/`.
The script `data/download_training_data.sh` downloads the
datasets and places them in the correct directory if ran from
`data/` directory (it uses the `osf` command line tool
[from osfclient](https://github.com/osfclient/osfclient)),
but this can be done manually as well.

The model filters learned with the code in `training_scripts` are 
provided in `data/trained_models/`. Training the filters for
one task with the training parameters used here takes about 2
hours on the CPU of a modern laptop.

Using the provided filters and downloading the data,
the code in `plotting_scripts` can be used to reproduce the
figures in the paper.

## Running the code

If the data is in place, the scripts in `plotting_scripts` can
be run from main directory to reproduce the figures in the paper.

The scripts for learning the filters in `training_scripts` can
be run from the main directory as well.

The scripts have a description of what they do at the
beginning of the file. The names of the scripts are
also self-explanatory.

The scripts for synthesizing the datasets in `stimulus_synthesis`
should be run from the directory in which they are located.
The scripts `analysis1_CL_finding.m` and
`analysis2_BV_generation_depth_motion.m` should be ran in
that order. The script `params1_stimulus_generation.m` has
the stimulus synthesis parameters. The directory in
`code/stimulus_synthesis/matlab_functions/` should
be added to the MATLAB path. This code uses the Matlab
[BurgeLab toolbox](https://jburge.psych.upenn.edu/code.html),
which assumes that the dataset with natural images described in
the paper is in the directory
`/Users/Shared/VisionScience/Project_Databases/LRSI/`.

## Dependencies

TO DO

