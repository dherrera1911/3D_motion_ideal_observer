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
#### LOAD AND TIDY STIMULUS DATASET
##############

# specify what trained model to load
savePlots = True
dnK = 2
spd = '0.15'
degStep = '7.5'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00' # Trained model disparity variability
dspStdVec = ['00', '02', '05', '10', '15'] # Disparity variability datasets to test

# Specify parameters of how to generate the figures
summaryCtg = 22.5 # Direction of motion to summarize effect of disparity
interpPoints = 11 # Number of interpolation points between categories
samplesPerStim = 10 # Number of noise samples for stimulus initialization
# Specify the indices of different filter subsets
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
lrFiltersInd = np.array([0, 1, 2, 3, 4])
bfFiltersInd = np.array([5, 6, 7, 8, 9])
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

# Make the directory to save the plots for this model
plotDirName = f'./plots/3D_dir/dnK{dnK}_spd{spd}_noise{noise}_' + \
    f'degStep{degStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# Put different filters indices in a list
filterInds = [allFiltersInd, lrFiltersInd, bfFiltersInd]
filterType = ['All', 'LR', 'BF']

##############
#### LOAD TRAINED FILTERS
##############

# LOAD THE TRAINED MODEL FILTERS
modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))
# Initialize random AMA model
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']

###############
# GET MODEL PERFORMANCE FOR EACH DISPARITY-VARIABILITY DATASET
###############


plotTypeDirName = f'{plotDirName}7_performance_periphery/'
os.makedirs(plotTypeDirName, exist_ok=True)
plt.rcParams.update({'font.size': 18, 'font.family': 'Nimbus Sans'})

# Get the LR component of category to summarize
summaryCtgLR = np.sin(np.deg2rad(summaryCtg)) * np.double(spd)
summaryCtgLR = np.round(summaryCtgLR * 10000) / 10000
# Dictionary to store performance for summary category
bfSummary = {'All':[], 'LR':[], 'BF':[]}
lrSummary = {'All':[], 'LR':[], 'BF':[]}

# If we're not adapting statistics, load original dataset
if not adaptStats:
    ## LOAD STIMULUS DATASET TO GET CATEGORIES
    # Training
    data = spio.loadmat('./data/ama_inputs/direction_looming/'
      f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
      f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
    s, ctgInd, ctgVal = unpack_matlab_data(
        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
    # Get only angle of motion
    ctgVal = ctgVal[1,:]
    nCtg = len(ctgVal)
    # Get a vector of categories that is shifted, to match literature convention
    ctgVal360 = shift_angles(ctgVal)

# Get the estimates
for dInd in range(len(dspStdVec)):
    dspStd = dspStdVec[dInd]
    # Load the dataset with this disparity std
    data = spio.loadmat('./data/ama_inputs/direction_looming_dspVar/'
      f'D3D-nStim_0300-spd_{spd}-degStep_{degStep}-'
      f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
    sPer, ctgIndPer, ctgValPer = unpack_matlab_data(
        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
    # Use only angle of motion for ctgVal
    ctgValPer = ctgValPer[1,:]
    ctgValPerShift = shift_angles(ctgValPer) # Shift categories to match literature convention
    # Get the cagetgory for Left right components
    ctgValLR = torch.sin(torch.deg2rad(ctgValPer)) * 0.15
    ctgValLR = torch.round(ctgValLR * 10000) / 10000
    # Convert intensity stimuli to contrast stimuli
    sPer = contrast_stim(s=sPer, nChannels=2)
    # Get performance statistics
    lrStats = {}
    bfStats ={}
    for ft in range(len(filterType)):
        ### INITIALIZE AMA MODEL TO GET ESTIMATES
        # Extract the name and indices of the filters
        tName = filterType[ft]
        fInds = filterInds[ft]
        # Initialize AMA model with random filters
        if adaptStats:
            ama = init_trained_ama(amaDict=trainingDict, sAll=sPer, ctgInd=ctgIndPer,
                                   ctgVal=ctgValPer, samplesPerStim=samplesPerStim)
        else:
            ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                                   ctgVal=ctgVal, samplesPerStim=samplesPerStim)
        # Assign the learned filters to the AMA object
        ama.assign_filter_values(fNew=trainingDict['filters'][fInds,:])
        ama.update_response_statistics()
        # Interpolate class statistics
        ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

        ### OBTAIN THE ESTIMATES
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

        ### PROCESS ESTIMATES, AND COLLAPSE CATEGORIES WITH SAME LR COMPONENT
        # Get the back-foth and lr components of the categories
        bfCtg = torch.sign(torch.cos(torch.deg2rad(ctgValPer)))
        bfCtg[torch.abs(ctgValPer)==90] = 0
        lrCtg = torch.sin(torch.deg2rad(ctgValPer)) * 0.15
        # Get towards-away sign estimates
        bfSign = torch.sign(torch.cos(torch.deg2rad(estimates)))
        # Find estimates that are frontoparallel plane, to remove from statistics
        tol = 0 # Tolerance for frontoparallel plane, in degrees
        absEstimate = torch.abs(estimates)
        # get non-frontoparallel estimates indices to remove later
        notFpInds = torch.where((absEstimate > 90+tol) | (absEstimate < 90-tol))
        # Get whether the estimate has the correct bf sign
        bfSignAgree = (bfSign == bfCtg[ctgIndReps])
        bfSignAgree = bfSignAgree.type(torch.float)
        # Get the left-right component
        leftRightComp = torch.sin(torch.deg2rad(estimates)) * 0.15
        # Convert to LR error
        lrError = torch.abs(leftRightComp - lrCtg[ctgIndReps])
        # Get statistics on the estimates
        bfStats[tName] = au.get_estimate_statistics(estimates=bfSignAgree[notFpInds],
                                                    ctgInd=ctgIndReps[notFpInds])
        lrStats[tName] = au.get_estimate_statistics(estimates=lrError, ctgInd=ctgIndReps)
        # Group categories by LR speed
        groupedCtg = torch.unique(ctgValLR)
        nGrCtg = len(groupedCtg)
        bfStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
        lrStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
        for n in range(nGrCtg):
            inds = torch.where(ctgValLR == groupedCtg[n])[0]
            bfStats[tName]['meanGrouped'][n] = torch.mean(bfStats[tName]['estimateMean'][inds])
            lrStats[tName]['meanGrouped'][n] = torch.mean(lrStats[tName]['estimateMean'][inds])

        ### SAVE RESULT FOR THE CATEGORY TO PLOT VS DISPARITY VARIABILITY
        summaryInds = torch.isin(groupedCtg, torch.tensor([summaryCtgLR, -summaryCtgLR],
                                                          dtype=torch.float32))
        bfS = torch.mean(bfStats[tName]['meanGrouped'][summaryInds])
        lrS = torch.mean(lrStats[tName]['meanGrouped'][summaryInds])
        bfSummary[tName].append(bfS)
        lrSummary[tName].append(lrS)

    ##################
    # PLOT CI VS FRONTOPARALLEL COMPONENT FOR THIS DISPARITY VARIABILITY
    ##################
    dspStd = dspStdVec[dInd]
    ms = 7 # Marker size
    lw = 2 # Line width
    fs = 18 # Font size
    # PLOT BACK-FORTH ERROR ACROSS CATEGORIES FOR THIS DISPARITY VARIABILITY
    # remove the frontoparallel categories
    plotInds = torch.arange(1, nGrCtg-1)
    plt.plot(groupedCtg[plotInds], 1-bfStats['All']['meanGrouped'][plotInds],
             'black', marker='o', markersize=ms, linewidth=lw, label='All filters')
    plt.plot(groupedCtg[plotInds], 1-bfStats['LR']['meanGrouped'][plotInds],
             'tab:orange', marker='o', markersize=ms, linewidth=lw, label='FP filters')
    plt.plot(groupedCtg[plotInds], 1-bfStats['BF']['meanGrouped'][plotInds],
             'tab:purple', marker='o', markersize=ms, linewidth=lw, label='TA filters')
    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1])
    plt.xticks([-0.15, 0, 0.15])
    # Add legend
    plt.xlabel('Frontoparallel speed (m/s)', fontsize=fs)
    plt.ylabel('Towards-away confusions', fontsize=fs)
    plt.legend()
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.savefig(fname=f'{plotTypeDirName}towards_away_conf_respNoise_{addRespNoise}_'
                    f'noisyCov_{statsNoise}_dspStd_{dspStd}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
    # Plot left-right estimates
    plt.plot(groupedCtg, lrStats['All']['meanGrouped'], 'black', marker='o',
             markersize=ms, linewidth=lw, label='All filters')
    plt.plot(groupedCtg, lrStats['LR']['meanGrouped'], 'tab:orange', marker='o',
             markersize=ms , linewidth=lw, label='FP filters')
    plt.plot(groupedCtg, lrStats['BF']['meanGrouped'], 'tab:purple', marker='o',
             markersize=ms, linewidth=lw, label='TA filters')
    plt.ylim(0, 0.05)
    plt.yticks([0, 0.025, 0.05])
    plt.xticks([-0.15, 0, 0.15])
    plt.xlabel('Frontoparallel speed (m/s)', fontsize=fs)
    plt.ylabel('Frontoparallel MAE (m/s)', fontsize=fs)
    # Add legend
    plt.legend()
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.savefig(fname=f'{plotTypeDirName}bf_respNoise_{addRespNoise}_'
                    f'noisyCov_{statsNoise}_dspStd_{dspStd}.png',
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


# Convert disparity std to float for x axis plotting
dspStdPlt = [float(x) for x in dspStdVec]

# Plot back-forth error as a function of disparity std
plt.rcParams.update({'font.size': 20, 'font.family': 'Nimbus Sans'})
ms = 9
lw = 3

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
fig.set_size_inches(5, 4)
plt.legend(loc='lower right', fontsize=18)
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
fig.set_size_inches(5.1, 4)
plt.ylim(0,0.04)
plt.yticks([0, 0.02, 0.04], fontsize=18)
plt.savefig(fname=f'{plotTypeDirName}Summary_lr_mae_respNoise_{addRespNoise}_'
            f'noisyCov_{statsNoise}.png',
            bbox_inches='tight', pad_inches=0.1)
plt.close()


