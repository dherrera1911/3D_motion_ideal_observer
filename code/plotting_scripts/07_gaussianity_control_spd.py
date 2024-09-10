##########################################
# This script tests the 3D-speed model using simulated responses
# sampled from Gaussian distributions. This is a control
# to test the effect of non-Gaussianity of the responses
# on the results.
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
import torch.distributions.multivariate_normal as mvn
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
loom = '1'
dspStd = '00'
plotDirName = f'./plots/gaussianity_test/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
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

# Initialize ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                       ctgVal=ctgVal, samplesPerStim=samplesPerStim)


# Make a testing dataset of responses with gaussian distribution
repeats = 5
respTstGauss = []
ctgIndGauss = []
for ctg in range(nCtg):
    # How many stimuli per category
    nSamples = torch.sum(ctgIndTst==ctg)
    ctgIndGauss.append(torch.ones(nSamples * repeats) * ctg)
    # Make gaussian distribution
    mean = ama.respMean[ctg,:].detach().clone()
    cov = ama.respCov[ctg,:,:].detach().clone()
    dist = mvn.MultivariateNormal(loc=mean,
                                  covariance_matrix=cov)
    # Take samples
    X = dist.sample([nSamples*repeats])
    respTstGauss.append(X)
respTstGauss = torch.cat(respTstGauss)
ctgIndGauss = torch.cat(ctgIndGauss)
ctgIndGauss = ctgIndGauss.long()



###############
# 0) PLOT DETERMINANT OF COVARIANCES
###############

# Plot determinant of covariance matrix for the subsets of filters
plotTypeDirName = f'{plotDirName}0_covariance_determinant/'
os.makedirs(plotTypeDirName, exist_ok=True)
for k in range(len(filterIndList)):
    # Initialize ama model
    ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                           ctgVal=ctgVal, samplesPerStim=samplesPerStim)
    # Select filter subset
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    respSubset = respTstGauss[:,filterIndList[k]]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Get determinant of covariance matrices
    det = torch.linalg.slogdet(ama.respCov.detach())
    plt.plot(ctgVal, det[1], color=filterColor[k], label=filterType[k],
             lw=4, ms=12, marker='o')
plt.xlabel('3D speed (m/s)')
plt.ylabel('Log determinant of covariance matrix')
plt.legend()

plt.savefig(fname=f'{plotTypeDirName}estimates_density_{filterType[k]}.png',
      bbox_inches='tight', pad_inches=0)
plt.close()


###############
# 1) GET MODEL ESTIMATES WITH FILTER SUBSETS
###############

plotTypeDirName = f'{plotDirName}1_estimates_speed/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Trim the edges that have border effects in estimation
inds2plot = torch.arange(ctgTrim, nCtg-ctgTrim)
estimates = []
ctgIndList = []
statsList = []

plt.rcParams.update({'font.size': 28, 'font.family': 'Nimbus Sans'})

for k in range(len(filterIndList)):
    # Initialize ama model
    ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                           ctgVal=ctgVal, samplesPerStim=samplesPerStim)
    # Select filter subset
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    respSubset = respTstGauss[:,filterIndList[k]]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama = interpolate_ama(ama, interpPoints=interpPoints)
    # Get estimates
    ll = ama.resp_2_ll(resp=respSubset)
    post = ama.ll_2_posterior(ll=ll)
    estimates = ama.posterior_2_estimate(posteriors=post,
                                            method4est='MAP').detach()
    # Get statistics of estimates
    statsList.append(au.get_estimate_statistics(estimates=estimates,
                                                ctgInd=ctgIndGauss,
                                                quantiles=[0.16, 0.84]))
    errorInterval = torch.cat((statsList[k]['lowCI'].reshape(-1,1),
            statsList[k]['highCI'].reshape(-1,1)), dim=-1).transpose(0,1)
    ##########
    # Plot the estimates median and CI
    ##########
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
    ctgIndPlot = ctgIndGauss[torch.isin(ctgIndGauss, inds2plot)]
    estPlot = estimates[torch.isin(ctgIndGauss, inds2plot)]
    ##########
    # Plot scatter of individual estimates 
    ##########
    jitter = torch.rand(len(ctgIndPlot)) * 0.05 - 0.025
    sns.scatterplot(x=ctgVal[ctgIndPlot]+jitter, y=estPlot,
                    color=filterColor[k], alpha=0.1)
    sns.scatterplot(x=ctgVal[inds2plot],
                    y=statsList[k]['estimateMedian'][inds2plot],
                    color=filterColor[k], s=30) 
    plt.xlabel('3D speed (m/s)')
    plt.ylabel('3D speed estimates (m/s)')
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
# 2) FILTER SUBSETS CONFIDENCE INTERVALS
###############

plotTypeDirName = f'{plotDirName}2_confidence_intervals_speed/'
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
    # Initialize ama model
    ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                           ctgVal=ctgVal, samplesPerStim=samplesPerStim)
    # Select filter subset
    filterSubset = trainingDict['filters'][fInds]
    respSubset = respTstGauss[:,fInds]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    if not statsNoise:
        ama.respCov = ama.respCovNoiseless.detach()
    ama = interpolate_ama(ama, interpPoints=interpPoints)
    # Get estimates
    ll = ama.resp_2_ll(resp=respSubset)
    post = ama.ll_2_posterior(ll=ll)
    estimates = ama.posterior_2_estimate(posteriors=post,
                                            method4est='MAP').detach()
    # Compute estimate statistics
    statsSubtype[tName] = au.get_estimate_statistics(
        estimates=estimates, ctgInd=ctgIndGauss, quantiles=quantiles)
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


