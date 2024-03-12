##########################################
# This script plots the main results for the 3D speed task.
# It plots both example preprocessed stimuli, the filters
# learned by the model, and the performance of the model
# at 3D speed estimation with different filter subsets.
# It also plots some other figures not included in the paper
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
#### LOAD DATA
##############

# SPECIFY WHAT STIMULUS DATASET AND MODEL TO LOAD
savePlots = True
dnK = 2
spdStep = '0.100'
maxSpd = '2.50'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '0'
dspStd = '00'
plotDirName = f'./plots/3D_speed/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
    f'spdStep{spdStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# SPECIFY THE INDICES OF DIFFERENT FILTER SUBSETS
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
monoFiltersInd = np.array([0, 1, 2, 3])
binoFiltersInd = np.array([4, 5, 6, 7, 8, 9])

# SPECIFY INTERPOLATION AND SUBSAMPLING PARAMETERS
interpPoints = 11 # Number of interpolation points between categories
ctg2plot = torch.tensor([9, 13, 17, 21, 23, 25]) # Indices of categories to plot
ctgTrim = 5 # Number of categories to trim from the edges for plotting without border effects
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

# PROCESS STIMULUS DATASET
# Convert indices and categories to Z-motion speeds
ctgVal = polar_2_Z(ctgVal)
ctgVal, ctgInd = au.sort_categories(ctgVal=ctgVal, ctgInd=ctgInd)
ctgValTst = polar_2_Z(ctgValTst)
ctgValTst, ctgIndTst = au.sort_categories(ctgVal=ctgValTst, ctgInd=ctgIndTst)
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
nCtg = len(ctgVal)
# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

# Put different filters indices in a list
filterIndList = [allFiltersInd, monoFiltersInd, binoFiltersInd]
filterType = ['All', 'Mono', 'Bino']
filterColor = ['k', 'r', 'b']

##############
#### INITIALIZE TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_' \
    f'spdStep_{spdStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Initialize random AMA model
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']
ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=10, respNoiseVar=respNoiseVar,
        pixelCov=pixelNoiseVar, ctgVal=ctgVal,
        samplesPerStim=samplesPerStim, nChannels=2)
# Assign the learned filter values to the model
ama.assign_filter_values(fNew=trainingDict['filters'])
ama.update_response_statistics()

##############
#### PLOT MODEL OUTPUTS
##############

###############
# 0) PLOT AND EXPORT NOISY NORMALIZED STIMULI
###############

plotTypeDirName = f'{plotDirName}0_noisy_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)

rawIms = torch.tensor(data.get('Iccd').transpose())

nStimPlot = 3
for k in range(nCtg):
    # Step 1: filter rows by category
    indices = (ctgInd == k).nonzero(as_tuple=True)[0]
    s_k = s[indices]
    rawIms_k = rawIms[indices]
    # random sample of rows
    sampleIndices = torch.randperm(s_k.shape[0])[:nStimPlot]
    sSample = s_k[sampleIndices]
    rawSample = rawIms_k[sampleIndices]
    # Step 2: apply the function to generate noisy samples
    sNoisy = ama.preprocess(s=sSample)
    # Step 3: convert the 2D array sNoisy with shape (nStimPlot, df), into a 3D array
    sNoisy = unvectorize_1D_binocular_video(sNoisy, nFrames=nFrames)
    sNoiseless = unvectorize_1D_binocular_video(sSample, nFrames=nFrames)
    sRaw = unvectorize_1D_binocular_video(rawSample, nFrames=nFrames)
    # Step 4: plot and save each matrix from the new sNoisy
    for i in range(sNoisy.shape[0]):
        stimList = [sRaw, sNoiseless, sNoisy]
        stimName = ['raw', 'noiseless', 'noisy']
        for j in range(3):
            if j==0:
                nPix = int(df/nFrames/2)
                diffEyes = stimList[j][i,0,0] - stimList[j][i,0,-1]
                stimList[j][i,:,0:nPix] = stimList[j][i,:,0:nPix].clone() - diffEyes
            plt.imshow(stimList[j][i,:,:].squeeze(), cmap='gray')
            ax = plt.gca()
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if savePlots:
                spdStr = f'{ctgVal[k]:.2f}'
                fileName = f'{plotTypeDirName}{stimName[j]}_spd{spdStr}_sample{i}_type{j}.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
            plt.close()

###############
# 1) PLOT FILTERS
###############

plotTypeDirName = f'{plotDirName}1_filters/'
os.makedirs(plotTypeDirName, exist_ok=True)

fAll = ama.all_filters().detach()
nFilters = fAll.shape[0]
nPairs = int(nFilters/2)
fAll2D = unvectorize_1D_binocular_video(fAll, nFrames=nFrames)

if savePlots:
    for k in range(nFilters):
        plt.imshow(fAll2D[k, :, :].squeeze(), cmap='gray')
        ax = plt.gca()
        plt.axis('off')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        fileName = f'{plotTypeDirName}filter{k+1}.png'
        plt.savefig(fname=fileName, bbox_inches='tight', pad_inches=0)
        plt.close()

###############
# 2) PLOT RESPONSE ELLIPSES
###############

plotTypeDirName = f'{plotDirName}3_covariances/'
os.makedirs(plotTypeDirName, exist_ok=True)

addRespNoise = False # Add response noise to the model
nPairs = int(ama.f.shape[0]/2)
responses = ama.get_responses(s=s, addRespNoise=addRespNoise).detach()
if addRespNoise:
    respCov = ama.respCov.clone().detach()
else:
    respCov = ama.respCovNoiseless.clone().detach()

# Subsample stimuli for better visualization
sSubs = 2 # Subsample factor
respSub = responses[::sSubs,:]
ctgIndSub = ctgInd[::sSubs]
# Subsample classes for better visualization
ctgSubs = 6 # Subsample factor
indKeep = subsample_categories_centered(nCtg=nCtg, subsampleFactor=ctgSubs)
indRemove = np.arange(nCtg)[~np.isin(np.arange(nCtg), indKeep)]
# Get the subsampled stimuli and categories
ctgValSub, ctgIndSub, respSub = remove_categories(removeCtg=indRemove,
        ctgVal=ctgVal, ctgInd=ctgIndSub, s=respSub)
respMeanSub = ama.respMean.detach()[indKeep,:]
# Get the covariance matrices to plot
if addRespNoise:
    respCovSub = ama.respCov.detach()[indKeep,:,:]
else:
    respCovSub = ama.respCovNoiseless.detach()[indKeep,:,:]

# Formatting
cmap = sns.diverging_palette(220, 20, s=80, l=70, sep=1, center="dark", as_cmap=True)
plt.rcParams.update({'font.size': 30, 'font.family': 'Nimbus Sans'})
# Choose filter pairs to plot scatter responses
fPairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0,2], [1,3], [1,2], [0,3]]
for n in range(len(fPairs)):
    fInd = fPairs[n]
    pairCov = subsample_cov_inds(covariance=respCovSub, keepInds=fInd)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    # Plot the responses
    ap.response_scatter(ax=ax, resp=respSub[:,fInd], ctgVal=ctgValSub[ctgIndSub],
                        colorMap=cmap)
    ap.plot_ellipse_set(mean=respMeanSub[:,fInd], cov=pairCov,
                        ctgVal=ctgValSub, colorMap=cmap, ax=ax)
    plt.xlabel(f'f{fInd[0]+1} response')
    plt.ylabel(f'f{fInd[1]+1} response')
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ap.add_colorbar(ax=ax, ctgVal=ctgValSub, colorMap=cmap,
                     label='  Speed \n (m/s)', ticks=[-2, -1, 0, 1, 2])
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}f{fInd[0]+1}f{fInd[1]+1}_ellipses_noise{addRespNoise}.png',
              bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

# Plot covariance lines 
# Make the subplots axes
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))
ap.plot_covariance_values(axes=ax, covariance=respCov, color='black',
                          xVal=ctgVal, size=2)
# Set tick font size
for a in range(len(fig.axes)):
    fig.axes[a].tick_params(axis='both', which='major', labelsize=10)
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}covariances_noisy{addRespNoise}.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


# Plot interpolated covariance lines
#covInterp = covariance_interpolation(covariance=respCov, nPoints=interpPoints)
#xInterp = spline_interpolation(y=ctgVal, nPoints=interpPoints)
## Make the subplots axes
#fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
#ap.plot_covariance_values(axes=ax, covariance=covInterp, color='black',
#                          xVal=xInterp, size=2)
#ap.plot_covariance_values(axes=ax, covariance=respCov, color='red',
#                          xVal=ctgVal, size=4)
#plt.show()


###############
# 3) GET MODEL ESTIMATES WITH FILTER SUBSETS
###############

plotTypeDirName = f'{plotDirName}2_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Trim the edges that have border effects in estimation
inds2plot = torch.arange(ctgTrim, nCtg-ctgTrim)
estimates = []
ctgIndList = []
statsList = []
repeats = 5

plt.rcParams.update({'font.size': 28, 'font.family': 'Nimbus Sans'})

for k in range(len(filterIndList)):
    # Select filter subset
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                           nPoints=interpPoints)
    ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                      nPoints=interpPoints)
    ama.ctgVal = torch.tensor(linear_interpolation(y=ctgVal, nPoints=interpPoints),
                              dtype=torch.float32)

    # Get estimates, with multiple noise samples
    estTemp = []
    ctgIndTemp = []
    for r in range(repeats):
        print('Repeat: ', r)
        estTemp.append(ama.get_estimates(s=sTst, method4est='MAP',
                                           addRespNoise=True).detach())
        ctgIndTemp.append(ctgIndTst)

    # Tidy estimates into lists
    estTemp = torch.tensor(torch.cat(estTemp), dtype=torch.float32)
    ctgIndTemp = torch.cat(ctgIndTemp)
    estimates.append(estTemp)
    ctgIndList.append(ctgIndTemp)
    # Get statistics of estimates
    statsList.append(au.get_estimate_statistics(estimates=estimates[k],
                                                ctgInd=ctgIndList[k],
                                                quantiles=[0.16, 0.84]))
    errorInterval = torch.cat((statsList[k]['lowCI'].reshape(-1,1),
            statsList[k]['highCI'].reshape(-1,1)), dim=-1).transpose(0,1)
    # Plot the estimates and CI
    fig, ax = plt.subplots()
    plot_estimate_statistics(ax=ax, estMeans=statsList[k]['estimateMedian'][inds2plot],
        errorInterval=errorInterval[:,inds2plot], ctgVal=ctgVal[inds2plot], color=filterColor[k])
    plt.xlabel('3D speed (m/s)')
    plt.ylabel('3D speed estimates (m/s)')
    # Set tick font size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=24)
    # Save the plot
    if savePlots:
        fig, ax = plt.gcf(), plt.gca()
        fig.tight_layout(rect=[0, 0, 0.95, 0.95])
        fig.set_size_inches(8, 7)
        plt.savefig(fname=f'{plotTypeDirName}model_estimates_{filterType[k]}.png',
              bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

    # Get ctgInd in inds2plot
    ctgIndPlot = ctgIndList[k][torch.isin(ctgIndList[k], inds2plot)]
    estPlot = estimates[k][torch.isin(ctgIndList[k], inds2plot)]
    # Make density plot
    jitter = torch.rand(len(ctgIndPlot)) * 0.05 - 0.025
    sns.scatterplot(x=ctgVal[ctgIndPlot]+jitter, y=estPlot,
                    color=filterColor[k], alpha=0.1)
    sns.scatterplot(x=ctgVal[inds2plot],
                    y=statsList[k]['estimateMedian'][inds2plot],
                    color=filterColor[k], s=30) 
    # Set plot limits
    plt.ylim([ctgVal[inds2plot].min(), ctgVal[inds2plot].max()])
    plt.xlim([ctgVal[inds2plot].min(), ctgVal[inds2plot].max()])
    # Save the plot
    if savePlots:
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(8, 8)
        plt.savefig(fname=f'{plotTypeDirName}estimates_density_{filterType[k]}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


###############
# 4) FILTER SUBSETS CONFIDENCE INTERVALS
###############

plotTypeDirName = f'{plotDirName}4_confidence_intervals/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Put the different filters into a list to loop over and get estimates
nf = '10'

quantiles = [0.16, 0.84]
method4est = 'MAP'
repeats = 5
addRespNoise = True # Add response noise to the model
statsNoise = True # Use statistics that include response noise

plt.rcParams.update({'font.size': 30, 'font.family': 'Nimbus Sans'})
amaSubtype = {}
statsSubtype = {}

inds2plot = np.arange(start=ctgTrim, stop=nCtg-ctgTrim)
ctgValX = ctgVal[inds2plot]

# Get the confidence intervals
for ft in range(len(filterType)):
    # Extract the name and indices of the filters
    tName = filterType[ft]
    fInds = filterIndList[ft]
    # Make interpolated ama model
    # Select filter subset
    filterSubset = trainingDict['filters'][fInds]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    if statsNoise:
        ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                               nPoints=interpPoints)
    else:
        ama.respCov = covariance_interpolation(covariance=ama.respCovNoiseless.detach(),
                                               nPoints=interpPoints)
    ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                      nPoints=interpPoints)
    ama.ctgVal = torch.tensor(linear_interpolation(y=ctgVal, nPoints=interpPoints),
                              dtype=torch.float32)
    # Obtain the estimates
    estimates = []
    ctgIndList = []
    # Loop over repeats
    for r in range(repeats):
        print('Repeat: ', r)
        estimates.append(ama.get_estimates(s=sTst, method4est=method4est,
                                           addRespNoise=addRespNoise).detach())
        ctgIndList.append(ctgIndTst)
    estimates = torch.cat(estimates)
    ctgIndReps = torch.cat(ctgIndList)
    # Compute estimate statistics
    statsSubtype[tName] = au.get_estimate_statistics(
        estimates=estimates, ctgInd=ctgIndReps, quantiles=quantiles)
    statsSubtype[tName]['ciWidth'] = statsSubtype[tName]['highCI'] - \
        statsSubtype[tName]['lowCI']
    # Remove statistics of categories that show border effect
    statsSubtype[tName] = remove_categories_stats(statsDict=statsSubtype[tName],
                                           inds2keep=inds2plot)

# Plot the confidence intervals
plotQuant = 'ciWidth'   #'ciWidth', 'estimateSD
# Plot confidence intervals
plt.plot(ctgValX, statsSubtype['All'][plotQuant], 'black', lw=4, ms=12, marker='o', label='All')
plt.plot(ctgValX, statsSubtype['Mono'][plotQuant], 'red', lw=4, ms=12, marker='o', label='Monocular')
plt.plot(ctgValX, statsSubtype['Bino'][plotQuant], 'blue', lw=4, ms=12, marker='o', label='Binocular')

plt.yscale('log')
yticks = [0.125, 0.25, 0.5, 1, 2]
plt.yticks(yticks, [str(ytick) for ytick in yticks])
xticks = [-2, -1, 0, 1, 2]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
plt.xlabel('3D speed (m/s)')
plt.ylabel('68% CI width (m/s)')
plt.legend()
# Put legend in top center within the plot
plt.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', borderaxespad=0.,
           fontsize=24)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=28)
#plt.ylim((0.01, 5))
if savePlots:
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(14, 7)
    plt.savefig(fname=f'{plotTypeDirName}{method4est}_respNoise_{addRespNoise}_'
                f'noisyCov_{statsNoise}_nFilt_{nf}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


###############
# 5) PLOT POSTERIORS
###############

plotTypeDirName = f'{plotDirName}5_posteriors/'
os.makedirs(plotTypeDirName, exist_ok=True)


for k in range(len(filterIndList)):
    # Make interpolated ama model
    # Select filter subset
    filterSubset = trainingDict['filters'][fInds]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    if statsNoise:
        ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                               nPoints=interpPoints)
    else:
        ama.respCov = covariance_interpolation(covariance=ama.respCovNoiseless.detach(),
                                               nPoints=interpPoints)
    ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                      nPoints=interpPoints)
    ama.ctgVal = torch.tensor(linear_interpolation(y=ctgVal, nPoints=interpPoints),
                              dtype=torch.float32)
    # Compute posteriors
    posteriors = ama.get_posteriors(s=sTst).detach()
    # Find the interpolated category indices and values
    ctgValInterp = ama.ctgVal
    ctgIndInterp = find_interp_indices(ctgVal, ctgValInterp, ctgIndTst)
    ctg2plotInterp = (ctg2plot) * (interpPoints + 1)
    # Initialize figure with subplots
    nCols = 3
    nRows = 2
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(12,6))
    for i in range(len(ctg2plot)):
        inds = ctgIndInterp == ctg2plotInterp[i]
        postCtg = posteriors[inds,:]
        # Plot the posteriors
        ap.plot_posterior(ax=ax.flatten()[i], posteriors=postCtg,
                  ctgVal=ctgValInterp, trueVal=ctgVal[ctg2plot[i]])
        # Set axes title
        ax.flatten()[i].set_title(f'{ctgVal[ctg2plot[i]]:.1f} m/s')
        # If it is the first row, remove x ticks
        if i < nCols:
            ax.flatten()[i].set_xticks([])
        else:
            ax.flatten()[i].set_xlabel('Speed (m/s)')
        # If it is first column, set y label
        if i % nCols == 0:
            ax.flatten()[i].set_ylabel('Posterior probability')
        # Remove y ticks
        ax.flatten()[i].set_yticks([])
    # Set figure size
    fig.set_size_inches(13, 3)
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}posteriors{filterType[k]}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(12,6))
    for i in range(len(ctg2plot)):
        # Class posteriors
        posteriorCtg = posteriors[:, ctg2plotInterp[i]]
        # Plot response of likelihood neurons
        ap.plot_posterior_neuron(ax=ax.flatten()[i],
                                 posteriorCtg=posteriorCtg,
                                 ctgInd=ctgIndTst, ctgVal=ctgVal,
                                 trueVal=ctgVal[ctg2plot[i]])
        # If it is the first row, remove x ticks
        if i < nCols:
            ax.flatten()[i].set_xticks([])
        else:
            ax.flatten()[i].set_xlabel('Speed (m/s)')
        # If it is first column, set y label
        if i % nCols == 0:
            ax.flatten()[i].set_ylabel('Likelihood neuron response')
        # Remove y ticks
        ax.flatten()[i].set_yticks([])
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}likelihood_neurons_{filterType[k]}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

