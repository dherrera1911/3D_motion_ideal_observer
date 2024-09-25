# Download files from OSF repository

mkdir data/ama_inputs

# 3D speed, with looming
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/S3D-nStim_0500-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TRN.mat data/ama_inputs/S3D-nStim_0500-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TRN.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TST.mat data/ama_inputs/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TST.mat

# 3D direction, with looming
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/D3D-nStim_0500-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TRN.mat data/ama_inputs/D3D-nStim_0500-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TRN.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TST.mat data/ama_inputs/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TST.mat

# 3D speed with disparity variability
mkdir data/ama_inputs/speed_disparity_variability

osf -p w9mpe fetch osfstorage/AMA_input_stimuli/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TST.mat data/ama_inputs/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_02-dnK_2-loom_1-TST.mat data/ama_inputs/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_02-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_05-dnK_2-loom_1-TST.mat data/ama_inputs/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_05-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_10-dnK_2-loom_1-TST.mat data/ama_inputs/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_10-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_15-dnK_2-loom_1-TST.mat data/ama_inputs/speed_disparity_variability/S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_15-dnK_2-loom_1-TST.mat

# 3D direction with disparity variability
mkdir data/ama_inputs/direction_disparity_variability

osf -p w9mpe fetch osfstorage/AMA_input_stimuli/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TST.mat data/ama_inputs/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_00-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_02-dnK_2-loom_1-TST.mat data/ama_inputs/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_02-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_05-dnK_2-loom_1-TST.mat data/ama_inputs/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_05-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_10-dnK_2-loom_1-TST.mat data/ama_inputs/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_10-dnK_2-loom_1-TST.mat
osf -p w9mpe fetch osfstorage/AMA_input_stimuli/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_15-dnK_2-loom_1-TST.mat data/ama_inputs/direction_disparity_variability/D3D-nStim_0300-spd_0.15-degStep_7.5-dspStd_15-dnK_2-loom_1-TST.mat

