##########################################
# This script analyzes the contribution of different binocular cues
# (IOVD and CDOT) to the 3D speed task, by generating
# cue-isolating stimuli and testing the previously trained model
# on these stimuli.
# 
# Code author: Daniel Herrera-Esposito, dherrera1911 at gmail dot com
##########################################


##############
#### IMPORT PACKAGES
##############

import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
import seaborn as sns
import sys
sys.path.append('./code/')
from auxiliary_functions import *
import copy
import os

##############
#### SPECIFY WHAT MODEL TO LOAD
##############

savePlots = True
dnK = 2
spdStep = '0.100'
maxSpd = '2.50'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00'
interpPoints = 11 # Number of interpolation points between categories
samplesPerStim = 10 # Number of noise samples for stimulus initialization

# Specify the indices of different filter subsets
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
monoFiltersInd = np.array([0, 1, 2, 3])
binoFiltersInd = np.array([4, 5, 6, 7, 8, 9])

filterType = ['All', 'Mono', 'Bino']
filterColor = ['k', 'r', 'b']
filterIndList = [allFiltersInd, monoFiltersInd, binoFiltersInd]
stimColors = ['k', 'm', 'y', 'g']


# Directory to save the plots
plotDirName = f'./plots/3D_speed_cues/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
    f'spdStep{spdStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# Function to get error interval
def error_int(lowCI, highCI):
    errorInt = torch.cat((lowCI.reshape(-1,1),
                          highCI.reshape(-1,1)), dim=-1).transpose(0,1)
    return errorInt

##########
# INITIALIZE THE TRAINED AMA MODEL
##########

torch.no_grad()
modelFile = f'./data/trained_models/' \
    f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_' \
    f'spdStep_{spdStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Load training dataset
data = spio.loadmat('./data/ama_inputs/speed_looming/'
  f'S3D-nStim_0500-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
s = contrast_stim(s=s, nChannels=2)
# Convert indices and categories to Z-motion speeds
ctgVal = polar_2_Z(ctgVal)
ctgVal, ctgInd = au.sort_categories(ctgVal=ctgVal, ctgInd=ctgInd)
# Extract some properties of the dataset
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
nCtg = len(ctgVal)

##########
# LOAD TESTING DATASET
##########

### Make uncorrelated iovd dataset
dataTst = spio.loadmat('./data/ama_inputs/'
  f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Convert intensity stimuli to contrast stimuli
sTst = contrast_stim(s=sTst, nChannels=2)
ctgValTst = polar_2_Z(ctgValTst)
ctgValTst, ctgIndTst = au.sort_categories(ctgVal=ctgValTst, ctgInd=ctgIndTst)

# Pre-normalize stimuli for better mached stimuli
sNrm = au.normalize_stimuli_channels(s=sTst, nChannels=2)
# Reshape the testing stimuli into videos
sTst2D = unvectorize_1D_binocular_video(inputVec=sNrm, nFrames=nFrames)


##############
#### 1) GENERATE CUE-ISOLATING STIMULI (UNCORRELATED BETWEEN EYES)
##############

### Make IOVD dataset by matching stimuli of different eyes, keeping
# the video of a given eye the same
iovdUn = torch.empty_like(sTst2D)
monoPixels = int(df/(2*nFrames))
for ctg in range(nCtg):
    stimIndices = (ctgIndTst == ctg).nonzero(as_tuple=True)[0]
    # We'll randomly permute the stimulus indices so that for each frame
    # we choose a different stimulus
    stimIndicesRandom = stimIndices[torch.randperm(len(stimIndices))]
    # For each frame, we will choose a frame from another stimulus in the original dataset
    iovdUn[stimIndices, :, :monoPixels] = sTst2D[stimIndices, :, :monoPixels]
    iovdUn[stimIndices, :, monoPixels:] = sTst2D[stimIndicesRandom, :, monoPixels:]
# Turn iovd into vector
iovdUn = vectorize_2D_binocular_video(iovdUn)

### Make Anticorrelated IOVD dataset by reversing polarity of one eye
iovdRev = sTst2D.clone()
monoPixels = int(df/(2*nFrames))
# Randomly select indives to invert left and right eye
randi = torch.randint(low=0, high=2, size=(iovdRev.shape[0],))
indsL, = np.where(randi)
indsR, = np.where(-(1-randi))
# Invert polarity of each eye where corresponds
iovdRev[indsL,:,:30] = -iovdRev[indsL,:,:30]
iovdRev[indsR,:,30:] = -iovdRev[indsR,:,30:]
# Turn iovd into vector
iovdRev = vectorize_2D_binocular_video(iovdRev)


### Make CDOT dataset by permutating the frames of the stimuli, keeping
# the two eyes together
cdot = torch.empty_like(sTst2D)
for ctg in range(nCtg):
    stimIndices = (ctgIndTst == ctg).nonzero(as_tuple=True)[0]
    # For each frame, we will choose a frame from another stimulus in the original dataset
    for frame_idx in range(sTst2D.size(1)):
        # We'll randomly permute the stimulus indices so that for each frame
        # we choose a different stimulus
        stimIndicesRandom = stimIndices[torch.randperm(len(stimIndices))]
        # Fill the new dataset
        cdot[stimIndices, frame_idx, :] = sTst2D[stimIndicesRandom, frame_idx, :]
# Turn cdot into vector
cdot = vectorize_2D_binocular_video(cdot)


# Load CDOT dataset with appropriate retinal processing
#dataCDOT = spio.loadmat('./data/ama_inputs/speed_cdot/'
#  f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
#  f'dspStd_00-dnK_{dnK}-loom_{loom}-TST.mat')
#cdotRet, ctgIndCDOT, ctgValCDOT = unpack_matlab_data(
#    matlabData=dataCDOT, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
#cdotRet = contrast_stim(s=cdotRet, nChannels=2)
#
#lala = unvectorize_1D_binocular_video(inputVec=cdotRet, nFrames=nFrames)

# Put the different testing stimuli datasets in a list
stimType = ['Original', 'IOVD_Un', 'IOVD_Rev', 'CDOT']
stimSets = [sNrm, iovdUn, iovdRev, cdot]

###############
# 1) PLOT EXAMPLES OF CUE-ISOLATING STIMULI
###############

plotTypeDirName = f'{plotDirName}0_noisy_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)
plotStim = False

nStimPlot = 2
ctg2plot = torch.arange(1, nCtg, 3)
if plotStim:
    for k in range(len(ctg2plot)):
        c = ctg2plot[k]
        indices = (ctgIndTst == c).nonzero(as_tuple=True)[0]
        # random sample of rows
        sampleIndices = torch.randperm(len(indices))[:nStimPlot]

        # Extract samples for each type of stimulus
        samples = [stimSet[indices][sampleIndices] for stimSet in stimSets]
        # Convert samples into 2D matrices
        matrices = [unvectorize_1D_binocular_video(sample, nFrames=nFrames) for sample in samples]

        # Plot and save each matrix
        for i, stimMatrix in enumerate(zip(*matrices)):
            for j, (type_name, matrix) in enumerate(zip(stimType, stimMatrix)):
                plt.imshow(matrix.squeeze(), cmap='gray')
                ax = plt.gca()
                plt.axis('off')
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                if savePlots:
                    fileName = f'{plotTypeDirName}{type_name.lower()}_stim_spd{ctgVal[c]:.1f}_sample{i}.png'
                    plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()


###############
# 2) GET MODEL ESTIMATES
###############

plotTypeDirName = f'{plotDirName}2_cues_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

quantiles = torch.tensor([0.16, 0.84], dtype=torch.float32)

adaptStats = False # if True, use statistics of original stimuli. If false, us tested stimuli
addRespNoise = True
repeats = 5

# Get vector with only the speeds of the categories
ctgValSpd = torch.abs(ctgVal)

ctgTrim = 5  # Categories to not plot
inds2plot = torch.arange(ctgTrim, nCtg-ctgTrim) # inds to plot
stats = {}
directionAccuracy = {}
for k in range(len(filterType)):
    fInds = filterIndList[k]
    fnam = filterType[k]
    stats[fnam] = {}
    directionAccuracy[fnam] = {}
    for ss in range(len(stimSets)):
        stimName = stimType[ss]
        # Initialize ama model
        if adaptStats:
            sInit = stimSets[ss]
            ctgIndInit = ctgIndTst
        else:
            sInit = s
            ctgIndInit = ctgInd
        ama = init_trained_ama(amaDict=trainingDict, sAll=sInit, ctgInd=ctgIndInit,
                               ctgVal=ctgVal, samplesPerStim=samplesPerStim)
        # Assign the learned filter values to the model
        filterSubset = trainingDict['filters'][filterIndList[k],:]
        ama.assign_filter_values(fNew=filterSubset)
        ama.update_response_statistics()
        # Interpolate class statistics
        ama = interpolate_ama(ama, interpPoints)

        # GET MODEL ESTIMATES
        # Loop over repeats
        estimates = []
        ctgIndList = []
        for r in range(repeats):
            print('Repeat: ', r)
            estimates.append(ama.get_estimates(s=stimSets[ss], method4est='MAP',
                                                  addRespNoise=addRespNoise).detach())
            ctgIndList.append(ctgIndTst)
        estimates = torch.cat(estimates)
        ctgIndReps = torch.cat(ctgIndList)
        stats[fnam][stimName] = au.get_estimate_statistics(estimates=estimates,
                                                           ctgInd=ctgIndReps)
        # Make error intervals
        stats[fnam][stimName]['eInt'] = error_int(stats[fnam][stimName]['lowCI'],
                                                  stats[fnam][stimName]['highCI'])
        stats[fnam][stimName]['ciWidth'] = stats[fnam][stimName]['highCI'] - \
            stats[fnam][stimName]['lowCI']

        # GET ACCURACY OF DIRECTION OF MOTION
        estimateSign = torch.sign(estimates)
        trueSign = torch.sign(ctgValTst[ctgIndReps])
        correctSign = (estimateSign == trueSign).float()

        groupedCtg = torch.unique(ctgValSpd)
        nGrCtg = len(groupedCtg)
        directionAccuracy[fnam][stimName] = torch.zeros(nGrCtg)
        for n in range(nGrCtg):
            ctgIndSpd = torch.where(ctgValSpd == groupedCtg[n])[0]
            inds = torch.where(torch.isin(ctgIndReps, ctgIndSpd))[0]
            directionAccuracy[fnam][stimName][n] = correctSign[inds].mean()

        # PLOT THE ESTIMATES FOR THESE FILTERS WITH THESE STIMULI
#        plt.figure(figsize=(4,4))
#        plt.plot(ctgVal[inds2plot], stats[fnam][stimName]['estimateMedian'][inds2plot],
#                 color=filterColor[k])
#        plt.fill_between(ctgVal[inds2plot], stats[fnam][stimName]['lowCI'][inds2plot],
#                                    stats[fnam][stimName]['highCI'][inds2plot],
#                         color=filterColor[k], alpha=0.2)
#        plt.xlabel('Speed (m/s)')
#        plt.ylabel('Estimate (m/s)')
#        plt.title(f'{fnam} - {stimName}')
#        plt.savefig(fname=f'{plotTypeDirName}1_estimates_{fnam}_{stimName}_'
#        f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
#                    pad_inches=0)
#        plt.close()
#
        ##################
        # PLOT SCATTER OF ESTIMATES
        ##################
#        # Subsample estimates
#        subsFactor = 3
#        estimatesSubs = estimates[0:sTst.shape[0]]
#        estimatesSubs = estimatesSubs[torch.isin(ctgIndTst, inds2plot)]
#        ctgIndSubs = ctgIndTst[torch.isin(ctgIndTst, inds2plot)]
#        estimatesSubs = estimatesSubs[::subsFactor]
#        ctgIndSubs = ctgIndSubs[::subsFactor]
#        jitter = torch.rand(len(ctgIndSubs)) * 0.05 - 0.025
#        sns.scatterplot(x=ctgValTst[ctgIndSubs]+jitter, y=estimatesSubs,
#                        color=filterColor[k], alpha=0.1)
#        plt.xlabel('3D speed (m/s)')
#        plt.ylabel('3D speed estimates (m/s)')
#        # Set plot limits
#        plt.ylim([ctgValTst[inds2plot].min(), ctgValTst[inds2plot].max()])
#        plt.xlim([ctgValTst[inds2plot].min(), ctgValTst[inds2plot].max()])
#        fig, ax = plt.gcf(), plt.gca()
#        fig.set_size_inches(4, 4)
#        plt.title(f'{fnam} - {stimName}')
#        plt.savefig(fname=f'{plotTypeDirName}2_density_{fnam}_{stimName}_'
#        f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
#                    pad_inches=0)
#        plt.close()
#
    ###############
    # PLOT CI WIDTHS FOR DIFFERENT STIMULI FOR EACH FILTER TYPE
    ###############
#    plt.figure(figsize=(6,5))
#    plt.rcParams.update({'font.size': 18})
#    for ss in range(len(stimSets)):
#        stimName = stimType[ss]
#        # Plot estimates of all filters
#        plt.plot(ctgVal[inds2plot], stats[fnam][stimName]['ciWidth'][inds2plot],
#                 label=stimName, color=stimColors[ss], marker='o')
#    plt.yscale('log')
#    yticks = [0.25, 0.5, 1, 2, 4]
#    plt.ylim(0.1, 5)
#    plt.yticks(yticks, [str(ytick) for ytick in yticks])
#    plt.minorticks_off()
#    plt.xlabel('Speed (m/s)')
#    plt.ylabel('68% CI width (m/s)')
#    # Put legend outside of plot
#    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#    if savePlots:
#        plt.savefig(fname=f'{plotTypeDirName}3_CI_width_{fnam}_adaptedStats{adaptStats}_'
#        f'respNoise{addRespNoise}.png', bbox_inches='tight', pad_inches=0.1)
#        plt.close()
#    else:
#        plt.show()
#

# PLOT FOR THE DIFFERENT STIMULI, THE PERFORMANCE OF EACH FILTER
# IN DIRECTION SIGN

for ss in range(len(stimType)):
    stimName = stimType[ss]
    plt.figure(figsize=(6,5))
    plt.rcParams.update({'font.size': 18})
    for k in range(len(filterType)):
        fnam = filterType[k]
        plt.plot(groupedCtg[1:], directionAccuracy[fnam][stimType[ss]][1:],
                 label=fnam, color=filterColor[k], marker='o', ms=8)
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Direction discrimination accuracy')
    plt.ylim([0.5, 1])
    plt.legend()
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}3_direction_accuracy2_{stimName}_{adaptStats}_'
        f'respNoise{addRespNoise}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


###############
# 3) TRY CDOT ONLY SLOW CATEGORIES
###############

plotTypeDirName = f'{plotDirName}2_cues_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Get indices of slow categories
spdThreshold = 0.5
indsFast = np.where(torch.abs(ctgVal) > spdThreshold)[0]

# Make dataset with slow categories only
# Original stimuli
ctgValSl, ctgIndSl, sSl = remove_categories(removeCtg=indsFast, ctgVal=ctgVal,
                                            ctgInd=ctgInd, s=s)
# CDOT stimuli
ctgValSlTst, ctgIndSlTst, cdotSl = remove_categories(removeCtg=indsFast,
                                                     ctgVal=ctgVal, ctgInd=ctgIndTst,
                                                     s=cdot)
# Control test stimuli
ctgValSlTst, ctgIndSlTst, sSlTst = remove_categories(removeCtg=indsFast,
                                                      ctgVal=ctgVal, ctgInd=ctgIndTst,
                                                      s=sTst)

### Test model in CDOT stimuli
adaptStats = False
addStimNoise = True
addRespNoise = True

# Get vector with only the speeds of the categories
ctgValSpd = torch.abs(ctgValSl)
repeats = 4

statsSl = {}
directionAccuracy = {}
for k in range(len(filterType)):
    fInds = filterIndList[k]
    fnam = filterType[k]
    # Initialize ama model
    if adaptStats:
        sInit = cdotSl
        ctgIndInit = ctgIndSlTst
    else:
        sInit = sSl
        ctgIndInit = ctgIndSl
    ama = init_trained_ama(amaDict=trainingDict, sAll=sInit, ctgInd=ctgIndInit,
                           ctgVal=ctgValSl, samplesPerStim=samplesPerStim)

    # Assign the learned filter values to the model
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama = interpolate_ama(ama, interpPoints)

    # GET MODEL ESTIMATES
    # Loop over repeats
    estimates = []
    ctgIndList = []
    for r in range(repeats):
        print('Repeat: ', r)
        estimates.append(ama.get_estimates(s=cdotSl, method4est='MAP',
                                            addRespNoise=addRespNoise).detach())
        ctgIndList.append(ctgIndSlTst)
    estimates = torch.cat(estimates)
    ctgIndReps = torch.cat(ctgIndList)
    statsSl[fnam] = au.get_estimate_statistics(estimates=estimates,
                                                       ctgInd=ctgIndReps)
    # Make error intervals
    statsSl[fnam]['eInt'] = error_int(statsSl[fnam]['lowCI'],
                                              statsSl[fnam]['highCI'])
    statsSl[fnam]['ciWidth'] = statsSl[fnam]['highCI'] - \
        statsSl[fnam]['lowCI']

    # GET ACCURACY OF DIRECTION OF MOTION
    estimateSign = torch.sign(estimates)
    trueSign = torch.sign(ctgValSl[ctgIndReps])
    correctSign = (estimateSign == trueSign).float()

    groupedCtg = torch.unique(ctgValSpd)
    nGrCtg = len(groupedCtg)
    directionAccuracy[fnam] = torch.zeros(nGrCtg)
    for n in range(nGrCtg):
        ctgIndSpd = torch.where(ctgValSpd == groupedCtg[n])[0]
        inds = torch.where(torch.isin(ctgIndReps, ctgIndSpd))[0]
        directionAccuracy[fnam][n] = correctSign[inds].mean()


    # PLOT THE ESTIMATES FOR THESE FILTERS WITH THESE STIMULI
#    plt.figure(figsize=(4,4))
#    plt.plot(ctgValSl, statsSl[fnam]['estimateMedian'], color=filterColor[k])
#    #plt.plot(ctgValSl, statsSl[fnam]['estimateMean'], color=filterColor[k])
#    plt.fill_between(ctgValSl, statsSl[fnam]['lowCI'], statsSl[fnam]['highCI'],
#                     color=filterColor[k], alpha=0.2)
#    plt.xlabel('Speed (m/s)')
#    plt.ylabel('Estimate (m/s)')
#    plt.title(f'{fnam} - slow CDOT')
#    plt.savefig(fname=f'{plotTypeDirName}4_estimates_{fnam}_slowCDOT_'
#    f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
#                pad_inches=0)
#    plt.close()
#
#    ##################
#    # PLOT SCATTER OF ESTIMATES
#    ##################
#    # Subsample estimates
#    subsFactor = 3
#    estimatesSubs = estimates[0:cdotSl.shape[0]]
#    estimatesSubs = estimatesSubs[::subsFactor]
#    ctgIndSubs = ctgIndSlTst[::subsFactor]
#    jitter = torch.rand(len(ctgIndSubs)) * 0.05 - 0.025
#    sns.scatterplot(x=ctgValSl[ctgIndSubs]+jitter, y=estimatesSubs,
#                    color=filterColor[k], alpha=0.1)
#    plt.xlabel('3D speed (m/s)')
#    plt.ylabel('3D speed estimates (m/s)')
#    # Set plot limits
#    plt.ylim([ctgValSlTst.min(), ctgValSlTst.max()])
#    plt.xlim([ctgValSlTst.min(), ctgValSlTst.max()])
#    fig, ax = plt.gcf(), plt.gca()
#    fig.set_size_inches(4, 4)
#    plt.title(f'{fnam} - slow CDOT')
#    plt.savefig(fname=f'{plotTypeDirName}4_density_{fnam}_slowCDOT_'
#    f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
#                pad_inches=0)
#    plt.close()


# PLOT FOR THE DIFFERENT STIMULI, THE PERFORMANCE OF EACH FILTER
# IN DIRECTION SIGN
plt.figure(figsize=(6,5))
for k in range(len(filterType)):
    fnam = filterType[k]
    plt.plot(groupedCtg[1:], directionAccuracy[fnam][1:],
             label=fnam, color=filterColor[k], marker='o', ms=8)
plt.xlabel('Speed (m/s)')
plt.ylabel('Direction discrimination accuracy')
plt.ylim([0.5, 1])
plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}3_direction_accuracy2_CDOT_slow_{adaptStats}_'
    f'respNoise{addRespNoise}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


### TEST MODEL WITH ORIGINAL STIMULI CONSTRAINED TO SLOW CATEGORIES
addStimNoise = True
addRespNoise = True

repeats = 5

statsSl = {}
directionAccuracy = {}

for k in range(len(filterType)):
    fInds = filterIndList[k]
    fnam = filterType[k]
    statsSl[fnam] = {}
    # Initialize ama model
    sInit = sSl
    ctgIndInit = ctgIndSl
    ama = init_trained_ama(amaDict=trainingDict, sAll=sInit, ctgInd=ctgIndInit,
                           ctgVal=ctgValSl, samplesPerStim=samplesPerStim)

    # Assign the learned filter values to the model
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama = interpolate_ama(ama, interpPoints)

    # GET MODEL ESTIMATES
    # Loop over repeats
    estimates = []
    ctgIndList = []
    for r in range(repeats):
        print('Repeat: ', r)
        estimates.append(ama.get_estimates(s=sSlTst, method4est='MAP',
                                            addRespNoise=addRespNoise).detach())
        ctgIndList.append(ctgIndSlTst)
    estimates = torch.cat(estimates)
    ctgIndReps = torch.cat(ctgIndList)
    statsSl[fnam] = au.get_estimate_statistics(estimates=estimates,
                                                       ctgInd=ctgIndReps)
    # Make error intervals
    statsSl[fnam]['eInt'] = error_int(statsSl[fnam]['lowCI'],
                                              statsSl[fnam]['highCI'])
    statsSl[fnam]['ciWidth'] = statsSl[fnam]['highCI'] - \
        statsSl[fnam]['lowCI']

    # GET ACCURACY OF DIRECTION OF MOTION
    estimateSign = torch.sign(estimates)
    trueSign = torch.sign(ctgValSl[ctgIndReps])
    correctSign = (estimateSign == trueSign).float()

    groupedCtg = torch.unique(ctgValSpd)
    nGrCtg = len(groupedCtg)
    directionAccuracy[fnam] = torch.zeros(nGrCtg)
    for n in range(nGrCtg):
        ctgIndSpd = torch.where(ctgValSpd == groupedCtg[n])[0]
        inds = torch.where(torch.isin(ctgIndReps, ctgIndSpd))[0]
        directionAccuracy[fnam][n] = correctSign[inds].mean()

    # Plot the estimates for these filters with these stimuli
    plt.figure(figsize=(4,4))
    plt.plot(ctgValSl, statsSl[fnam]['estimateMedian'], color=filterColor[k])
    #plt.plot(ctgValSl, statsSl[fnam]['estimateMean'], color=filterColor[k])
    plt.fill_between(ctgValSl, statsSl[fnam]['lowCI'], statsSl[fnam]['highCI'],
                     color=filterColor[k], alpha=0.2)
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Estimate (m/s)')
    plt.title(f'{fnam} - slow Original')
    plt.savefig(fname=f'{plotTypeDirName}5_estimates_{fnam}_slowOriginal_'
    f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
                pad_inches=0)
    plt.close()

    ##################
    # PLOT SCATTER OF ESTIMATES
    ##################
    # Subsample estimates
    subsFactor = 3
    estimatesSubs = estimates[0:sSlTst.shape[0]]
    estimatesSubs = estimatesSubs[::subsFactor]
    ctgIndSubs = ctgIndSlTst[::subsFactor]
    jitter = torch.rand(len(ctgIndSubs)) * 0.05 - 0.025
    sns.scatterplot(x=ctgValSl[ctgIndSubs]+jitter, y=estimatesSubs,
                    color=filterColor[k], alpha=0.1)
    plt.xlabel('3D speed (m/s)')
    plt.ylabel('3D speed estimates (m/s)')
    # Set plot limits
    plt.ylim([ctgValSlTst.min(), ctgValSlTst.max()])
    plt.xlim([ctgValSlTst.min(), ctgValSlTst.max()])
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(4, 4)
    plt.title(f'{fnam} - slow CDOT')
    plt.savefig(fname=f'{plotTypeDirName}4_density_{fnam}_slowCDOT_'
    f'adaptedStats{adaptStats}_respNoise{addRespNoise}.png', bbox_inches='tight',
                pad_inches=0)
    plt.close()

# PLOT FOR THE DIFFERENT STIMULI, THE PERFORMANCE OF EACH FILTER
# IN DIRECTION SIGN
plt.figure(figsize=(6,5))
for k in range(len(filterType)):
    fnam = filterType[k]
    plt.plot(groupedCtg[1:], directionAccuracy[fnam][1:],
             label=fnam, color=filterColor[k], marker='o', ms=8)
plt.xlabel('Speed (m/s)')
plt.ylabel('Direction discrimination accuracy')
plt.ylim([0.5, 1])
#plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}3_direction_accuracy2_Original_slow_{adaptStats}_'
    f'respNoise{addRespNoise}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()

