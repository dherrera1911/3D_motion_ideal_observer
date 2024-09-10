## Optimal motion-in-depth estimation with natural stimuli

This is the code associated with the paper
["Optimal motion-in-depth estimation with natural stimuli"](https://www.biorxiv.org/content/10.1101/2024.03.14.585059v1.abstract).

The code is organized in three main directories according to
three different tasks:
* `code/stimulus_synthesis/`: Generate naturalistic
binocular videos used in the experiments. Subdirectories
`make_speed_dataset` and `make_direction_dataset` generate the
two different tasks.
* `code/training_scripts/`: Training ideal observer model (AMA).
There's one script for each task.
* `code/plotting_scripts/`: Evaluating the trained AMA models and
generate the paper plots.

Stimuli datasets (synthesized with code in `stimulus_synthesis`)
can be downloaded from the companion [OSF repository](https://osf.io/w9mpe/).
The datasets should be placed in directory `data/ama_inputs/`.
The script `data/download_training_data.sh` downloads the
datasets and places them in the correct directory if ran from
`data/` directory (it uses the `osf` command line tool
[from osfclient](https://github.com/osfclient/osfclient)),
but this can be done manually as well.

The model filters trained with the code in `training_scripts` are 
provided in `data/ama_filters/`.




