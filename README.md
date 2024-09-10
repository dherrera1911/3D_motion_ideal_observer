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
Run the script `data/download_stimuli.sh` from the main directory
to download the datasets to the correct directory, or
download them manually from the OSF repository
(the script uses the package
[osfclient](https://github.com/osfclient/osfclient)).

Pre-trained model filters are provided in `data/trained_models/`.
Training the filters for one task with the training parameters used
here takes about 2 hours on the CPU of a modern laptop.

Using the provided filters and downloading the data,
the code in `plotting_scripts` can be used to reproduce the
figures in the paper.

## Running the code

If the data is in place, the scripts in `plotting_scripts` can
be run from main directory to reproduce the figures in the paper.

The scripts for learning the filters in `training_scripts` can
be run from the main directory as well (although this is
optional, since pre-trained filters are provided).

The scripts above have a description of what they do at the
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

The code for training the models and plotting the results
are written in Python 3. All of the analysis revolves around the
Python [AMA package](https://github.com/dherrera1911/accuracy_maximization_analysis).
All dependencies can be installed in a new conda environment
with the file `environment.yml` in the main directory.

The code for synthesizing the stimuli is written in Matlab.
The code uses the Matlab [BurgeLab toolbox](https://jburge.psych.upenn.edu/code.html).


