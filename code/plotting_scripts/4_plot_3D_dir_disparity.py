##########################################
# This script plots the effects of disparity variability on
# 3D direction estimation.
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
import seaborn as sns
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
import sys
sys.path.append('./code/')
from auxiliary_functions import *
import copy
import os


##############
#### SPECIFY PARAMETERS
##############

# SPECIFY WHAT STIMULUS DATASET AND MODEL TO LOAD
savePlots = True
dnK = 2
spd = '0.15'
degStep = '7.5'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '0'
dspStd = '00'
plotDirName = f'./plots/3D_dir/dnK{dnK}_spd{spd}_noise{noise}_' + \
    f'degStep{degStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# SPECIFY THE INDICES OF DIFFERENT FILTER SUBSETS
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
lrFiltersInd = np.array([0, 1, 2, 3, 4])
bfFiltersInd = np.array([5, 6, 7, 8, 9])
# Specify interpolation and subsampling parameters
# Estimation parameters
interpPoints = 11 # Number of interpolation points between categories
samplesPerStim = 10 # Number of noise samples for stimulus initialization

# LOAD STIMULUS DATASET TO GET CATEGORIES
# Training
data = spio.loadmat('./data/ama_inputs/'
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')

# Process stimulus dataset
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
nCtg = len(ctgVal)
# Get only angle of motion
ctgVal = ctgVal[1,:]
# Get a vector of categories that is shifted, to match literature convention
ctgVal360 = shift_angles(ctgVal)

# Put different filters indices in a list
filterInds = [allFiltersInd, lrFiltersInd, bfFiltersInd]
filterType = ['All', 'LR', 'BF']

# LOAD THE TRAINED MODEL FILTERS
modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))
# Initialize random AMA model
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']


###############
# PERFORMANCE WITH DISPARITY VARIABILITY
###############

plotTypeDirName = f'{plotDirName}7_performance_periphery/'
os.makedirs(plotTypeDirName, exist_ok=True)

dspStdVec = ['00', '02', '05', '10', '15']
dspStdPlt = [float(x) for x in dspStdVec]

# Parameters to make figure
nf = '10' #Number of filters to use
quantiles = [0.16, 0.84]
# Statistics interpolation parameters
interpPoints = 11
method4est = 'MAP'
repeats = 5
addRespNoise = True
statsNoise = True
adaptStats = True

plt.rcParams.update({'font.size': 30, 'font.family': 'Nimbus Sans'})

# Get the cagetgory for Left right components
ctgValLR = torch.sin(torch.deg2rad(ctgVal)) * 0.15
ctgValLR = torch.round(ctgValLR * 10000) / 10000

# Select direction to make summary
summaryCtg = 22.5
summaryCtgLR = np.sin(np.deg2rad(summaryCtg)) * np.double(spd)
summaryCtgLR = np.round(summaryCtgLR * 10000) / 10000

bfSummary = {'All':[], 'LR':[], 'BF':[]}
lrSummary = {'All':[], 'LR':[], 'BF':[]}

# Get estimate results for each disparity std, and
# each filter subset
for dInd in range(len(dspStdVec)):
    # LOAD THE STIMULUS DATASET FOR THIS DISPARITY STD
    dspStd = dspStdVec[dInd]
    data = spio.loadmat('./data/ama_inputs/'
      f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
      f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
    sPer, ctgIndPer, ctgValPer = unpack_matlab_data(
        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
    # Use only angle of motion for ctgVal
    ctgValPer = ctgValPer[1,:]
    ctgValPerShift = shift_angles(ctgValPer) # Shift categories to match literature convention
    # Convert intensity stimuli to contrast stimuli
    sPer = contrast_stim(s=sPer, nChannels=2) 
    # Get performance statistics
    lrStats = {}
    bfStats ={}
    for ft in range(len(filterType)):
        # Extract the name and indices of the filters
        tName = filterType[ft]
        fInds = filterInds[ft]
        # Initialize AMA model with random filters
        ama = cl.AMA_emp(sAll=sPer, ctgInd=ctgIndPer, nFilt=10, respNoiseVar=respNoiseVar,
                pixelCov=pixelNoiseVar, ctgVal=ctgVal, samplesPerStim=samplesPerStim, nChannels=2)
        # Assign the learned filters to the AMA object
        ama.assign_filter_values(fNew=trainingDict['filters'][fInds,:])
        ama.update_response_statistics()
        # Interpolate class statistics
        ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)
        # Obtain the estimates
        estimates = []
        ctgIndList = []
        # Loop over noise repeats
        for r in range(repeats):
            print('Repeat: ', r)
            estimates.append(ama.get_estimates(
              s=sPer, method4est=method4est, addRespNoise=addRespNoise).detach())
            ctgIndList.append(ctgIndPer)
        estimates = torch.cat(estimates)
        ctgIndReps = torch.cat(ctgIndList)
        # Get categories of LR and BF components
        bfCtg = torch.sign(torch.cos(torch.deg2rad(ctgVal)))
        bfCtg[torch.abs(ctgVal)==90] = 0
        lrCtg = torch.sin(torch.deg2rad(ctgVal)) * 0.15
        # Get the components of the estimates
        bfSign = torch.sign(torch.cos(torch.deg2rad(estimates)))
        # Find estimates within tol degrees of horizontal. Assign sign=0
        absEstimate = torch.abs(estimates)
        tol = 5
        fpEstimates = torch.where((absEstimate < 90+tol) & (absEstimate > 90-tol))
        bfSign[fpEstimates] = 0
        # Get the left-right component
        leftRightComp = torch.sin(torch.deg2rad(estimates)) * 0.15
        # Convert to LR error
        lrError = torch.abs(leftRightComp - lrCtg[ctgIndReps])
        # Get whether the estimate has right sign
        bfSignAgree = (bfSign == bfCtg[ctgIndReps])
        bfSignAgree = bfSignAgree.type(torch.float)
        # Get statistics on the components
        lrStats[tName] = au.get_estimate_statistics(estimates=lrError, ctgInd=ctgIndReps)
        bfStats[tName] = au.get_estimate_statistics(estimates=bfSignAgree, ctgInd=ctgIndReps)
        # Group categories by LR speed
        groupedCtg = torch.unique(ctgValLR)
        nGrCtg = len(groupedCtg)
        bfStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
        lrStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
        for n in range(nGrCtg):
            inds = torch.where(ctgValLR == groupedCtg[n])[0]
            bfStats[tName]['meanGrouped'][n] = torch.mean(bfStats[tName]['estimateMean'][inds])
            lrStats[tName]['meanGrouped'][n] = torch.mean(lrStats[tName]['estimateMean'][inds])
        # Save summary results
        summaryInds = torch.isin(groupedCtg, torch.tensor([summaryCtgLR, -summaryCtgLR],
                                                          dtype=torch.float32))
        bfS = torch.mean(bfStats[tName]['meanGrouped'][summaryInds])
        lrS = torch.mean(lrStats[tName]['meanGrouped'][summaryInds])
        bfSummary[tName].append(bfS)
        lrSummary[tName].append(lrS)

    # Plot the statistics for this disparity std (it's within the loop because
    # statistics are not saved)
    dspStd = dspStdVec[dInd]
    # Plot back-forth error
    plt.plot(groupedCtg, 1-bfStats['All']['meanGrouped'], 'black', marker='o',
             markersize=10 , linewidth=3, label='All filters')
    plt.plot(groupedCtg, 1-bfStats['LR']['meanGrouped'], 'tab:orange', marker='o',
             markersize=10, linewidth=3, label='FP filters')
    plt.plot(groupedCtg, 1-bfStats['BF']['meanGrouped'], 'tab:purple', marker='o',
             markersize=10, linewidth=3, label='TA filters')
    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1])
    plt.xticks([-0.15, 0, 0.15])
    # Add legend
    plt.xlabel('Frontoparallel speed (m/s)', fontsize=28)
    plt.ylabel('Towards-away confusions', fontsize=28)
    plt.legend()
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(7.5, 6)
        plt.savefig(fname=f'{plotTypeDirName}towards_away_conf_respNoise_{addRespNoise}_'
                    f'noisyCov_{statsNoise}_dspStd_{dspStd}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
    # Plot left-right estimates
    plt.plot(groupedCtg, lrStats['All']['meanGrouped'], 'black', marker='o',
             markersize=10, linewidth=3, label='All filters')
    plt.plot(groupedCtg, lrStats['LR']['meanGrouped'], 'tab:orange', marker='o',
             markersize=10 , linewidth=3, label='FP filters')
    plt.plot(groupedCtg, lrStats['BF']['meanGrouped'], 'tab:purple', marker='o',
             markersize=10, linewidth=3, label='TA filters')
    plt.ylim(0, 0.05)
    plt.yticks([0, 0.025, 0.05])
    plt.xticks([-0.15, 0, 0.15])
    plt.xlabel('Frontoparallel speed (m/s)', fontsize=28)
    plt.ylabel('Frontoparallel MAE (m/s)', fontsize=28)
    # Add legend
    plt.legend()
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(7.5, 6)
        plt.savefig(fname=f'{plotTypeDirName}bf_respNoise_{addRespNoise}_'
                    f'noisyCov_{statsNoise}_dspStd_{dspStd}.png',
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

# Plot back-forth error as a function of disparity std
plt.rcParams.update({'font.size': 22, 'font.family': 'Nimbus Sans'})
ms = 12
lw = 4
plt.plot(dspStdPlt, 1-torch.tensor(bfSummary['All']), 'black',
          marker='o', markersize=ms , linewidth=lw, label='All')
plt.plot(dspStdPlt, 1-torch.tensor(bfSummary['LR']), 'tab:orange',
         marker='o', markersize=ms, linewidth=lw, label='Frontoparallel')
plt.plot(dspStdPlt, 1-torch.tensor(bfSummary['BF']), 'tab:purple',
         marker='o', markersize=ms, linewidth=lw, label='Towards-away')
plt.xlabel('Disparity std (arc min)')
plt.ylabel('Towards-away confusions')
plt.ylim(0, 0.6)
plt.yticks([0, 0.25, 0.5], fontsize=20)
fig, ax = plt.gcf(), plt.gca()
fig.tight_layout(rect=[0, 0, 0.95, 0.95])
fig.set_size_inches(6, 5)
plt.legend(loc='lower right', fontsize=20)
plt.savefig(fname=f'{plotTypeDirName}Summary_bf_confusions_respNoise_{addRespNoise}_'
            f'noisyCov_{statsNoise}.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()


# Plot left-right error as a function of disparity std
plt.plot(dspStdPlt, lrSummary['All'], 'black',
          marker='o', markersize=ms , linewidth=lw, label='All')
plt.plot(dspStdPlt, lrSummary['LR'], 'tab:orange',
         marker='o', markersize=ms, linewidth=lw, label='FP')
plt.plot(dspStdPlt, lrSummary['BF'], 'tab:purple',
         marker='o', markersize=ms, linewidth=lw, label='TA')
plt.xlabel('Disparity std (arc min)')
plt.ylabel('Frontoparallel MAE (m/s)')
fig, ax = plt.gcf(), plt.gca()
fig.tight_layout(rect=[0, 0, 0.95, 0.95])
fig.set_size_inches(6.2, 5)
plt.ylim(0,0.04)
plt.yticks([0, 0.02, 0.04], fontsize=20)
plt.savefig(fname=f'{plotTypeDirName}Summary_lr_mae_respNoise_{addRespNoise}_'
            f'noisyCov_{statsNoise}.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()


