##########################################
# This script plots the effects of disparity variability on
# 3D speed estimation.
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
import seaborn as sns
import copy
import os

##############
#### LOAD AND TIDY STIMULUS DATASET
##############

# specify what trained model to load
savePlots = True
dnK = 2
spdStep = '0.100'
maxSpd = '2.50'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00' # Trained model disparity variability
dspStdVec = ['00', '02', '05', '10', '15'] # Disparity variability datasets to load

# Specify parameters of how to generate the figures
# What speeds are "slow" and "fast" in the plots
lowSpd = 0.2 # Low speed to extract CI width
highSpd = 1.5 # High speed to extract CI width
interpPoints = 11 # Number of interpolation points between categories
ctg2plot = torch.tensor([9, 13, 17, 21, 23, 25]) # Indices of categories to plot
ctgTrim = 5 # Number of categories to trim from the edges for plotting without border effects
samplesPerStim = 5 # Number of noise samples in model initialization
nf = '10' #Number of filters to use
quantiles = [0.16, 0.84] # Quantiles for confidence intervals
method4est = 'MAP' # Method to decode estimate from posterior
repeats = 5 # Number of repeats to estimate confidence intervals
addRespNoise = True # Add response noise to the model
statsNoise = True # Use statistics that include response noise
adaptStats = True # Adapt response statistics to each dataset

# Specify the indices of different filter subsets
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
monoFiltersInd = np.array([0, 1, 2, 3])
binoFiltersInd = np.array([4, 5, 6, 7, 8, 9])

# Make the directory to save the plots for this model
plotDirName = f'./plots/3D_speed/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
    f'spdStep{spdStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# Put different filters in a list
filterIndList = [allFiltersInd, monoFiltersInd, binoFiltersInd]
filterType = ['All', 'Mono', 'Bino']
filterColor = ['k', 'r', 'b']

##############
#### LOAD TRAINED FILTERS
##############
modelFile = f'./data/trained_models/' \
    f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_' \
    f'spdStep_{spdStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'

trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))
# Parameters to initialize models
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']

###############
# GET MODEL PERFORMANCE FOR EACH DISPARITY-VARIABILITY DATASET
###############

plotTypeDirName = f'{plotDirName}6_confidence_intervals_periphery/'
os.makedirs(plotTypeDirName, exist_ok=True)
plt.rcParams.update({'font.size': 18, 'font.family': 'Nimbus Sans'})

# Dictionary to save the performance across disparity stds for two speeds
lowSpdCi = {'All':[], 'Mono':[], 'Bino':[]}
highSpdCi = {'All':[], 'Mono':[], 'Bino':[]}

# If we're not adapting statistics, load original dataset
if not adaptStats:
    data = spio.loadmat('./data/ama_inputs/speed_looming/'
      f'S3D-nStim_0500-spdStep_{spdStep}-maxSpd_{maxSpd}-'
      f'dspStd_00-dnK_{dnK}-loom_{loom}-TRN.mat')
    s, ctgInd, ctgVal = unpack_matlab_data(
        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
    ctgVal = polar_2_Z(ctgVal)
    ctgVal, ctgInd = au.sort_categories(ctgVal=ctgVal, ctgInd=ctgInd)
    s = contrast_stim(s=s, nChannels=2)

# LOOP OVER DISPARITY STDS TO GET ESTIMATES
for dInd in range(len(dspStdVec)):
    dspStd = dspStdVec[dInd]
    # Load the dataset with this disparity std
    data = spio.loadmat('./data/ama_inputs/speed_looming_dspVar/'
        f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
        f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
    sPer, ctgIndPer, ctgValPer = unpack_matlab_data(
        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
    # Convert indices and categories to Z-motion speeds
    ctgValPer = polar_2_Z(ctgValPer)
    ctgValPer, ctgIndPer = au.sort_categories(ctgVal=ctgValPer, ctgInd=ctgIndPer)
    nCtg = len(ctgValPer)
    # Trim the edges that have border effects in estimation
    inds2plot = torch.arange(ctgTrim, nCtg-ctgTrim)
    # x-axis value for the precision vs speed plots
    ctgValX = ctgValPer[inds2plot]
    # Extract some properties of the dataset
    nStim = sPer.shape[0]
    # Convert intensity stimuli to contrast stimuli
    sPer = contrast_stim(s=sPer, nChannels=2)
    statsSubtype = {}

    # GET THE ESTIMATES FOR THE DIFFERENT TYPES OF FILTERS
    for ft in range(len(filterType)):
        # Extract the name and indices of the filters
        tName = filterType[ft]
        fInds = filterIndList[ft]
        # Initialize the AMA model with random filters
        if adaptStats:
            sInit = sPer
            ctgIndInit = ctgIndPer
        else:
            sInit = s
            ctgIndInit = ctgInd
        ama = init_trained_ama(amaDict=trainingDict, sAll=sInit, ctgInd=ctgIndInit,
                               ctgVal=ctgVal, samplesPerStim=samplesPerStim)
        # Assign the loaded to the AMA object
        filterSubset = trainingDict['filters'][fInds]
        ama.assign_filter_values(fNew=filterSubset)
        ama.update_response_statistics()
        # Interpolate class statistics
        ama = interpolate_ama(ama, interpPoints=interpPoints)
        # Obtain the estimates of the model for this dataset
        estimates = []
        ctgIndList = []
        # Loop over repeats
        for r in range(repeats):
            print('Repeat: ', r)
            estimates.append(ama.get_estimates(s=sPer, method4est=method4est,
                                                  addRespNoise=addRespNoise).detach())
            ctgIndList.append(ctgIndPer)
        estimates = torch.cat(estimates)
        ctgIndReps = torch.cat(ctgIndList)
        # Compute estimate statistics for this dataset
        statsSubtype[tName] = au.get_estimate_statistics(
            estimates=estimates, ctgInd=ctgIndReps, quantiles=quantiles)
        statsSubtype[tName]['ciWidth'] = statsSubtype[tName]['highCI'] - \
            statsSubtype[tName]['lowCI']
        # Remove statistics of categories that show border effect 
        statsSubtype[tName] = remove_categories_stats(statsDict=statsSubtype[tName],
                                               inds2keep=inds2plot)
        # Extract value of CI width for 2 speeds for this disparity std
        lowSpdCi[tName].append(statsSubtype[tName]['ciWidth'][ctgValX==lowSpd])
        highSpdCi[tName].append(statsSubtype[tName]['ciWidth'][ctgValX==highSpd])

        ##################
        # PLOT SCATTER OF ESTIMATES
        ##################
        # Subsample estimates
        subsFactor = 3
        estimatesSubs = estimates[0:sPer.shape[0]]
        estimatesSubs = estimatesSubs[torch.isin(ctgIndPer, inds2plot)]
        ctgIndSubs = ctgIndPer[torch.isin(ctgIndPer, inds2plot)]
        estimatesSubs = estimatesSubs[::subsFactor]
        ctgIndSubs = ctgIndSubs[::subsFactor]
        jitter = torch.rand(len(ctgIndSubs)) * 0.05 - 0.025
        sns.scatterplot(x=ctgValPer[ctgIndSubs]+jitter, y=estimatesSubs,
                        color=filterColor[ft], alpha=0.1)
        plt.xlabel('3D speed (m/s)')
        plt.ylabel('3D speed estimates (m/s)')
        # Set plot limits
        plt.ylim([ctgValPer[inds2plot].min(), ctgValPer[inds2plot].max()])
        plt.xlim([ctgValPer[inds2plot].min(), ctgValPer[inds2plot].max()])
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(8, 8)
        plt.savefig(fname=f'{plotTypeDirName}estimates_density_{filterType[ft]}_dspVar_'
                        f'{dspStdVec[dInd]}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()

    ##################
    # PLOT CI VS SPEED FOR THIS DISPARITY VARIABILITY
    ##################
    fig, ax = plt.subplots(figsize=(13,5))
    dspStd = dspStdVec[dInd]
    plotQuant = 'ciWidth'   #'ciWidth', 'estimateSD
    ax.plot(ctgValX, statsSubtype['All'][plotQuant], 'black', lw=3, ms=8,
            marker='o', label='All')
    ax.plot(ctgValX, statsSubtype['Mono'][plotQuant], 'red', lw=3, ms=8,
            marker='o', label='Mono')
    ax.plot(ctgValX, statsSubtype['Bino'][plotQuant], 'blue', lw=3, ms=8,
            marker='o', label='Bino')
    ax.set_yscale('log')
    yticks = [0.25, 0.5, 1, 2]
    ax.set_yticks(yticks, [str(ytick) for ytick in yticks])
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('68% CI width (m/s)')
    ax.legend()
    ax.set_ylim((0.15, 2.7))
    # Locate legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.ylim((0.01, 5))
    if savePlots:
        fig.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.savefig(fname=f'{plotTypeDirName}{method4est}_adaptStats_{adaptStats}_'
                    f'dspStd_{dspStd}_respNoise_{addRespNoise}_noisyCov_'
                    f'{statsNoise}_nFilt_{nf}.png',
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

    # Plot the estimates
    for ft in range(len(filterType)):
        fig, ax = plt.subplots(figsize=(6,6))
        fName = filterType[ft]
        errorInterval = torch.cat((statsSubtype[fName]['lowCI'].unsqueeze(0),
                                    statsSubtype[fName]['highCI'].unsqueeze(0)), dim=0)
        plot_estimate_statistics(ax=ax, estMeans=statsSubtype[fName]['estimateMedian'],
            errorInterval=errorInterval, ctgVal=ctgValX, color=filterColor[ft])
        plt.xlabel('3D speed (m/s)')
        plt.ylabel('3D speed estimates (m/s)')
        if savePlots:
            fig.set_size_inches(6.5, 6)
            plt.savefig(fname=f'{plotTypeDirName}estimates_{fName}_{method4est}_'
                        f'adaptStats_{adaptStats}_dspStd_{dspStd}_'
                        f'respNoise_{addRespNoise}_noisyCov_'
                        f'{statsNoise}_nFilt_{nf}.png',
                        bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()


##################
# PLOT CI VS DISPARITY VARIABILITY FOR TWO SPEEDS (SLOW AND FAST)
##################
plt.rcParams.update({'font.size': 20, 'font.family': 'Nimbus Sans'})
ms = 9
lw = 3
# Parameters to make figure
dspStdPlt = [float(x) for x in dspStdVec]

for s in range(2):
    style = '-'
    if s==0:
        pltCi = lowSpdCi
        pltName = 'slow'
    else:
        pltCi = highSpdCi
        pltName = 'fast'
    plt.plot(dspStdPlt, torch.cat(pltCi['All']), 'black', lw=lw, ms=ms,
             marker='o', label='All', ls=style)
    plt.plot(dspStdPlt, torch.cat(pltCi['Mono']), 'red', lw=lw, ms=ms,
             marker='o', label='Monocular', ls=style)
    plt.plot(dspStdPlt, torch.cat(pltCi['Bino']), 'blue', lw=lw, ms=ms,
             marker='o', label='Binocular', ls=style)
    plt.xlabel('Disparity std (arc min)')
    plt.ylabel('68% CI width (m/s)')
    plt.yscale('log')
    yticks = [0.125, 0.25, 0.5, 1, 2]
    plt.ylim((0.07, 2.7))
    plt.yticks(yticks, [str(ytick) for ytick in yticks], fontsize=20)
    plt.legend(loc='lower right', fontsize=18)
    fig, ax = plt.gcf(), plt.gca()
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.set_size_inches(5, 4)
    plt.savefig(fname=f'{plotTypeDirName}Summary_{pltName}_adaptStats_{adaptStats}_'
                f'respNoise_{addRespNoise}_noisyCov_'
                f'{statsNoise}_nFilt_{nf}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


## Plot the scatter of filter responses to the stimuli
#for dInd in range(len(dspStdVec)):
#    dspStd = dspStdVec[dInd]
#    # Load the dataset with this disparity std
#    data = spio.loadmat('./data/ama_inputs/speed_looming_dspVar/'
#        f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
#        f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
#    sPer, ctgIndPer, ctgValPer = unpack_matlab_data(
#        matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
#    # Preprocess the dataset
#    # Convert indices and categories to Z-motion speeds
#    ctgValPer = polar_2_Z(ctgValPer)
#    ctgValPer, ctgIndPer = au.sort_categories(ctgVal=ctgValPer, ctgInd=ctgIndPer)
#    # Extract some properties of the dataset
#    nStim = sPer.shape[0]
#    # Convert intensity stimuli to contrast stimuli
#    sPer = contrast_stim(s=sPer, nChannels=2)
#    statsSubtype = {}
#    inds2plot = np.arange(start=ctgTrim, stop=nCtg-ctgTrim)
#    # Get the estimates for the different types of filters
#    for ft in range(len(filterType)):
#        # Extract the name and indices of the filters
#        tName = filterType[ft]
#        fInds = filterIndList[ft]
#        # Initialize the AMA model with random filters
#        if adaptStats:
#            ama = cl.AMA_emp(sAll=sPer, ctgInd=ctgIndPer, nFilt=len(fInds),
#                    respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar,
#                    ctgVal=ctgValPer, samplesPerStim=samplesPerStim, nChannels=2)
#        else:
#            ama = cl.AMA_emp(sAll=sAll, ctgInd=ctgInd, nFilt=len(fInds),
#                    respNoiseVar=respNoiseVar, pixelCov=pixelNoiseVar,
#                    ctgVal=ctgVal, samplesPerStim=samplesPerStim, nChannels=2)
#        # Assign the learned filters to the AMA object
#        filterSubset = trainingDict['filters'][fInds]
#        ama.assign_filter_values(fNew=filterSubset)
#        ama.update_response_statistics()
#
#
