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
import sys
sys.path.append('./code/')
from auxiliary_functions import *
import copy
import os

##############
#### LOAD STIMULI
##############

savePlots = True
dnK = 2
spdStep = '0.100'
maxSpd = '2.50'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '0'
dspStd = '00'
plotDirName = f'./plots/3D_speed_cues/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
    f'spdStep{spdStep}_loom{loom}/'
modelFile = f'./data/trained_models/' \
    f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_' \
    f'spdStep_{spdStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'

torch.no_grad()
os.makedirs(plotDirName, exist_ok=True)

# SPECIFY THE INDICES OF DIFFERENT FILTER SUBSETS
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
monoFiltersInd = np.array([0, 1, 2, 3])
binoFiltersInd = np.array([4, 5, 6, 7, 8, 9])

# SPECIFY INTERPOLATION AND SUBSAMPLING PARAMETERS
interpPoints = 11 # Number of interpolation points between categories
samplesPerStim = 10 # Number of noise samples for stimulus initialization

# LOAD STIMULUS DATASET
# Training
data = spio.loadmat('./data/ama_inputs/'
  f'S3D-nStim_0500-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Testing
dataTst = spio.loadmat('./data/ama_inputs/'
  f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

# Convert indices and categories to Z-motion speeds
ctgVal = polar_2_Z(ctgVal)
ctgVal, ctgInd = au.sort_categories(ctgVal=ctgVal, ctgInd=ctgInd)
ctgValTst = polar_2_Z(ctgValTst)
ctgValTst, ctgIndTst = au.sort_categories(ctgVal=ctgValTst, ctgInd=ctgIndTst)
# Extract some properties of the dataset
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
nCtg = len(ctgVal)

#######################
# MAKE CUE-ISOLATING STIMULI FROM LOADED DATASET
#######################

sTst = torch.cat((s, sTst), dim=0)
ctgIndTst = torch.cat((ctgInd, ctgIndTst), dim=0)

# Pre-normalize stimuli for nicer CDOT stimuli
sNrm = au.normalize_stimuli_channels(s=sTst, nChannels=2)
# Reshape the testing stimuli into videos
sTst2D = unvectorize_1D_binocular_video(inputVec=sNrm, nFrames=nFrames)

# Make CDOT dataset by permuting the frames of the stimuli, keeping
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

# Make IOVD dataset by matching stimuli of different eyes, keeping
# the video of a given eye the same
iovd = torch.empty_like(sTst2D)
monoPixels = int(df/(2*nFrames))
for ctg in range(nCtg):
    stimIndices = (ctgIndTst == ctg).nonzero(as_tuple=True)[0]
    # We'll randomly permute the stimulus indices so that for each frame
    # we choose a different stimulus
    stimIndicesRandom = stimIndices[torch.randperm(len(stimIndices))]
    # For each frame, we will choose a frame from another stimulus in the original dataset
    iovd[stimIndices, :, :monoPixels] = sTst2D[stimIndices, :, :monoPixels]
    iovd[stimIndices, :, monoPixels:] = sTst2D[stimIndicesRandom, :, monoPixels:]
# Turn iovd into vector
iovd = vectorize_2D_binocular_video(iovd)


###############
# PLOT CUE-ISOLATING STIMULI
###############

plotTypeDirName = f'{plotDirName}0_noisy_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)

nStimPlot = 5
for k in range(nCtg):
    # Step 1: filter rows by category
    indices = (ctgInd == k).nonzero(as_tuple=True)[0]
    cdot_k = cdot[indices]
    iovd_k = iovd[indices]
    orig_k = sTst[indices]
    # random sample of rows
    sampleIndices = torch.randperm(cdot_k.shape[0])[:nStimPlot]
    cdotSample = cdot_k[sampleIndices]
    iovdSample = iovd_k[sampleIndices]
    origSample = orig_k[sampleIndices]
    # Step 2: apply the function to generate noisy samples
    cdot2D = unvectorize_1D_binocular_video(cdotSample, nFrames=nFrames)
    iovd2D = unvectorize_1D_binocular_video(iovdSample, nFrames=nFrames)
    orig2D = unvectorize_1D_binocular_video(origSample, nFrames=nFrames)
    # Step 4: plot and save each matrix from the new sNoisy
    for i in range(cdot2D.shape[0]):
        plt.imshow(cdot2D[i,:,:].squeeze(), cmap='gray')
        ax = plt.gca()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if savePlots:
            fileName = f'{plotTypeDirName}cdot_stim_spd{ctgVal[k]:.1f}_sample{i}.png'
            plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
            plt.close()
        plt.imshow(iovd2D[i,:,:].squeeze(), cmap='gray')
        ax = plt.gca()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if savePlots:
            fileName = f'{plotTypeDirName}iovd_stim_spd{ctgVal[k]}_sample{i}.png'
            plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
            plt.close()
        plt.imshow(orig2D[i,:,:].squeeze(), cmap='gray')
        ax = plt.gca()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if savePlots:
            fileName = f'{plotTypeDirName}orig_stim_spd{ctgVal[k]}_sample{i}.png'
            plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
            plt.close()


###############
# GET MODEL ESTIMATES
###############

def error_int(lowCI, highCI):
    errorInt = torch.cat((lowCI.reshape(-1,1),
                          highCI.reshape(-1,1)), dim=-1).transpose(0,1)
    return errorInt

plotTypeDirName = f'{plotDirName}2_cues_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

quantiles = torch.tensor([0.16, 0.84], dtype=torch.float32)

# Subsets of filters to try separately
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
monoFiltersInd = np.array([0, 1, 2, 3])
binoFiltersInd = np.array([4, 5, 6, 7, 8, 9])

filterType = ['All', 'Mono', 'Bino']
filterColor = ['k', 'r', 'b']
filterIndList = [allFiltersInd, monoFiltersInd, binoFiltersInd]
stimType = ['Original', 'IOVD', 'CDOT']
stimSets = [sNrm, iovd, cdot]

adaptStats = True
addStimNoise = True
respNoise = True
stats = {}

for nf in range(len(filterType)):
    fInds = filterIndList[nf]
    fnam = filterType[nf]
    stats[fnam] = {}
    for ss in range(len(stimSets)):
        stimName = stimType[ss]
        trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))
        respNoiseVar = trainingDict['respNoiseVar']
        pixelNoiseVar = trainingDict['pixelNoiseVar']
        if adaptStats:
            ama = cl.AMA_emp(sAll=stimSets[ss], ctgInd=ctgIndTst, nFilt=10,
                             respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar,
                             ctgVal=ctgVal, samplesPerStim=samplesPerStim, nChannels=2)
        else:
            ama = cl.AMA_emp(sAll=stimSets[0], ctgInd=ctgIndTst, nFilt=10,
                             respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar,
                             ctgVal=ctgVal, samplesPerStim=samplesPerStim, nChannels=2)
        # Assign the learned filter values to the model
        filterSubset = trainingDict['filters'][filterIndList[nf],:]
        ama.assign_filter_values(fNew=filterSubset)
        ama.update_response_statistics()
        # Interpolate class statistics
        ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                               nPoints=interpPoints)
        ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                          nPoints=interpPoints)
        ama.ctgVal = torch.tensor(linear_interpolation(y=ctgVal, nPoints=interpPoints),
                                  dtype=torch.float32)
        # Get model estimates
        estimates = ama.get_estimates(s=stimSets[ss], method4est='MAP',
                                             addRespNoise=respNoise)
        stats[fnam][stimName] = au.get_estimate_statistics(estimates=estimates,
                                                           ctgInd=ctgIndTst)
        # Make error intervals
        stats[fnam][stimName]['eInt'] = error_int(stats[fnam][stimName]['lowCI'],
                                                  stats[fnam][stimName]['highCI'])
        stats[fnam][stimName]['ciWidth'] = stats[fnam][stimName]['highCI'] - \
            stats[fnam][stimName]['lowCI']


###############
# PLOT ESTIMATES
###############

ctgTrim = 5  # Categories to not plot
i2p = torch.arange(ctgTrim, nCtg-ctgTrim) # inds to plot
# Plot median and error intervals
# Start figure size
plt.figure(figsize=(12,4))
plotQuant = 'estimateMedian' #
for nf in range(len(filterType)):
    fnam = filterType[nf]
    plt.subplot(1,3,nf+1)
    plt.title(fnam)
    for ss in range(len(stimSets)):
        stimName = stimType[ss]
        plt.plot(ctgVal[i2p],
                 stats[fnam][stimName][plotQuant][i2p], label=stimName)
        plt.fill_between(ctgVal[i2p], stats[fnam][stimName]['lowCI'][i2p],
                                    stats[fnam][stimName]['highCI'][i2p], alpha=0.2)
plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

# Plot IOVD in different format
plt.figure(figsize=(8,2.5))

for nf in range(len(filterType)-1):
    fnam = filterType[nf+1]
    plt.subplot(1,2,nf+1)
#    plt.title(fnam)
    # Plot estimates of all filters
    plt.plot(ctgVal[i2p], stats['All']['IOVD'][plotQuant][i2p],
             label='All', color='k')
    plt.fill_between(ctgVal[i2p], stats['All']['IOVD']['lowCI'][i2p],
                     stats['All']['IOVD']['highCI'][i2p],
                     alpha=0.3, color='k')
    plt.plot(ctgVal[i2p], ctgVal[i2p], 'k--')
    plt.xlabel('Speed (m/s)')
    # Plot estimates of filter subset
    plt.plot(ctgVal[i2p], stats[fnam]['IOVD'][plotQuant][i2p],
             label=fnam, color=filterColor[nf+1])
    plt.fill_between(ctgVal[i2p], stats[fnam]['IOVD']['lowCI'][i2p],
                     stats[fnam]['IOVD']['highCI'][i2p],
                     alpha=0.3, color=filterColor[nf+1])
    plt.plot(ctgVal[i2p], ctgVal[i2p], 'k--')
    plt.xlabel('Speed (m/s)')
    if nf == 0:
        plt.ylabel('Estimate (m/s)')
    plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_IOVD_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


# Plot IOVD confidence intervals
plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size': 18})
for nf in range(len(filterType)):
    fnam = filterType[nf]
    # Plot estimates of all filters
    plt.plot(ctgVal[i2p], stats[fnam]['IOVD']['ciWidth'][i2p],
             label=fnam, color=filterColor[nf], marker='o')
plt.yscale('log')
yticks = [0.5, 1, 2]
plt.ylim(0.4, 2.5)
plt.yticks(yticks, [str(ytick) for ytick in yticks])
plt.minorticks_off()
plt.xlabel('Speed (m/s)')
plt.ylabel('68% CI width (m/s)')
#    plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}CI_width_IOVD_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


###############
# TRY CODT ONLY SLOW CATEGORIES
###############

# Get indices of slow categories
spdThreshold = 0.6
indsFast = np.where(torch.abs(ctgVal) > spdThreshold)[0]
# Make dataset with slow categories only
ctgValSl, ctgIndSl, sSl = remove_categories(removeCtg=indsFast, ctgVal=ctgVal,
                                            ctgInd=ctgIndTst, s=sTst)
_, _, cdotSl = remove_categories(removeCtg=indsFast, ctgVal=ctgVal,
                                 ctgInd=ctgIndTst, s=cdot)
#_, _, iovdSl = remove_categories(removeCtg=indsFast, ctgVal=ctgVal,
#                                 ctgInd=ctgIndTst, s=iovd)
# Test model in CDOT stimuli
adaptStats = True
addStimNoise = True
respNoise = True

statsSl = {}
stimName = 'cdot'

# Get estimates
for nf in range(len(filterType)):
    fInds = filterIndList[nf]
    fnam = filterType[nf]
    statsSl[fnam] = {}
    # Choose what stimuli to use for statistics
    if adaptStats:
        sInit = sSl
    else:
        sInit = cdotSl
    # Parameters to initialize model with only slow categories
    ama = cl.AMA_emp(sAll=sSl, ctgInd=ctgIndSl, nFilt=10,
                     respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar,
                     ctgVal=ctgValSl, samplesPerStim=samplesPerStim, nChannels=2)
    # Assign the learned filter values to the model
    filterSubset = trainingDict['filters'][filterIndList[nf],:]
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                           nPoints=interpPoints)
    ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                      nPoints=interpPoints)
    ama.ctgVal = torch.tensor(linear_interpolation(y=ctgValSl, nPoints=interpPoints),
                              dtype=torch.float32)
    # Get model estimates
    estimates = ama.get_estimates(s=cdotSl, method4est='MAP',
                                  addRespNoise=respNoise)
    statsSl[fnam] = au.get_estimate_statistics(estimates=estimates,
                                                       ctgInd=ctgIndSl)
    # Make error intervals
    statsSl[fnam]['eInt'] = error_int(statsSl[fnam]['lowCI'],
                                              statsSl[fnam]['highCI'])
    # Average error
    estimatesError = (estimates - ctgValSl[ctgIndSl]) ** 2
    tempStats = au.get_estimate_statistics(estimates=estimatesError,
                                            ctgInd=ctgIndSl)
    statsSl[fnam]['error'] = tempStats['estimateMean']


# Plot median and error intervals
# Start figure size
ctgTrim = 2  # Categories to not plot
nCtgSl = len(ctgValSl)
i2p = torch.arange(ctgTrim, nCtgSl-ctgTrim) # inds to plot

plt.figure(figsize=(12,4))
plotQuant = 'estimateMedian' #
for nf in range(len(filterType)):
    fnam = filterType[nf]
    plt.subplot(1,3,nf+1)
    plt.title(fnam)
    plt.plot(ctgValSl, statsSl[fnam][plotQuant], label=stimName)
    plt.fill_between(ctgValSl, statsSl[fnam]['lowCI'],
                                statsSl[fnam]['highCI'], alpha=0.2)
    plt.xlabel('Speed (m/s')
    if nf == 0:
        plt.ylabel('Estimate (m/s)')
    # Draw diagonal line
    plt.plot(ctgValSl, ctgValSl, 'k--')
    plt.plot(ctgValSl[i2p], ctgValSl[i2p], 'k--')
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}_slow.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()


# Plot CDOT in different format
plt.figure(figsize=(8,2.5))
for nf in range(len(filterType)-1):
    fnam = filterType[nf+1]
    plt.subplot(1,2,nf+1)
#    plt.title(fnam)
    # Plot estimates of all filters
    plt.plot(ctgValSl[i2p], statsSl['All'][plotQuant][i2p],
             label='All', color='k')
    plt.fill_between(ctgValSl[i2p], statsSl['All']['lowCI'][i2p],
                     statsSl['All']['highCI'][i2p],
                     alpha=0.2, color='k')
    plt.xlabel('3D-Speed (m/s)')
    plt.plot(ctgValSl[i2p], ctgValSl[i2p], 'k--')
    # Plot estimates of filter subset
    plt.plot(ctgValSl[i2p], statsSl[fnam][plotQuant][i2p],
             label=fnam, color=filterColor[nf+1])
    plt.fill_between(ctgValSl[i2p], statsSl[fnam]['lowCI'][i2p],
                     statsSl[fnam]['highCI'][i2p],
                     alpha=0.2, color=filterColor[nf+1])
    plt.plot(ctgValSl[i2p], ctgValSl[i2p], 'k--')
    if nf==0:
        plt.ylabel('Estimate (m/s)')
    plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_CDOT_slow_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


# Plot CDOT error
plt.figure(figsize=(8,4))
for nf in range(len(filterType)):
    fnam = filterType[nf]
    # Plot estimates of all filters
    plt.plot(ctgValSl[i2p], statsSl[fnam]['error'][i2p],
             label=fnam, color=filterColor[nf], marker='o')
    plt.ylabel('MSE')
    plt.xlabel('3D-Speed (m/s)')
    plt.legend()
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}Error_CDOT_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()


# Plot median and error intervals
# Start figure size
ctgTrim = 2  # Categories to not plot
nCtgSl = len(ctgValSl)
i2p = torch.arange(ctgTrim, nCtgSl-ctgTrim) # inds to plot
plt.figure(figsize=(6,5))
plotQuant = 'estimateMedian' #
plt.rcParams.update({'font.size': 18})
for nf in range(len(filterType)):
    fnam = filterType[nf]
    plt.plot(ctgValSl, statsSl[fnam][plotQuant], label=stimName,
             color=filterColor[nf], marker='o')
    plt.xlabel('Speed (m/s')
    if nf == 0:
        plt.ylabel('Estimate (m/s)')
# Draw diagonal line
plt.plot(ctgValSl, ctgValSl, 'k--')
# Set ticks
yticks = [-0.5, 0, 0.5]
plt.yticks(yticks, [str(ytick) for ytick in yticks])
xticks = [-0.5, 0, 0.5]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}slow_estimates_adaptedStats{adaptStats}_'
    f'stimNoise{addStimNoise}_respNoise{respNoise}_y_{plotQuant}.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


