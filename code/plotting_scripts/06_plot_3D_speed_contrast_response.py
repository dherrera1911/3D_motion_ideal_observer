##########################################
# This script plots the 3D speed estimation model's outputs for
# different stimulus contrasts. It manipulates the contrasts of
# both the datasets and of single stimuli and tests the model on
# these new stimuli.
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
from matplotlib.colors import Normalize
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
import sys
sys.path.append('./code/')
from auxiliary_functions import *
import seaborn as sns
import copy
import einops as eo
import os

savePlots = True
dnK = 2
spdStep = '0.100'
maxSpd = '2.50'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00'
plotDirName = f'./plots/3D_speed_contrast/dnK{dnK}_spd{maxSpd}_noise{noise}_' + \
    f'spdStep{spdStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

##############
#### LOAD STIMULI
##############
# TRAINING
data = spio.loadmat('./data/ama_inputs/'
  f'S3D-nStim_0500-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# TESTING
dataTst = spio.loadmat('./data/ama_inputs/'
  f'S3D-nStim_0300-spdStep_{spdStep}-maxSpd_{maxSpd}-'
  f'dspStd_00-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')

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

# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

##############
#### INITIALIZE TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_' \
    f'spdStep_{spdStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'

trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Initialize random AMA model
samplesPerStim = 5
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']

##############
#### PLOT MODEL OUTPUTS TO DIFFERENT CONTRASTS
##############

contrasts = [1, 0.5, 0.25, 0.125, 0.0625]

for c in range(len(contrasts)):
    # Initialize model
    ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=10, respNoiseVar=respNoiseVar,
            pixelCov=pixelNoiseVar, ctgVal=ctgVal,
            samplesPerStim=samplesPerStim, nChannels=2)

    ama.assign_filter_values(fNew=trainingDict['filters'])
    ama.update_response_statistics()

    # Apply contrast multiplier to dataset
    ctr = contrasts[c]
    sPlt = sTst.clone()*ctr

    ###############
    # 1) PLOT RESPONSE ELLIPSES
    ###############
    plotTypeDirName = f'{plotDirName}1_covariances/'
    os.makedirs(plotTypeDirName, exist_ok=True)

    addRespNoise = False
    nPairs = int(ama.f.shape[0]/2)
    responses = ama.get_responses(s=sPlt, addRespNoise=addRespNoise).detach()
    if addRespNoise:
        respCov = ama.respCov.clone().detach()
    else:
        respCov = ama.respCovNoiseless.clone().detach()

    # Subsample stimuli for better visualization
    sSubs = 1
    respSub = responses[::sSubs,:]
    ctgIndSub = ctgIndTst[::sSubs]
    # Subsample classes for better visualization
    ctgSubs = 6
    indKeep = subsample_categories_centered(nCtg=nCtg, subsampleFactor=ctgSubs)
    indRemove = np.arange(nCtg)[~np.isin(np.arange(nCtg), indKeep)]
    ctgValSub, ctgIndSub, respSub = remove_categories(removeCtg=indRemove,
            ctgVal=ctgVal, ctgInd=ctgIndSub, s=respSub)
    respMeanSub = ama.respMean.detach()[indKeep,:]
    if addRespNoise:
        respCovSub = ama.respCov.detach()[indKeep,:,:]
    else:
        respCovSub = ama.respCovNoiseless.detach()[indKeep,:,:]

    cmap = sns.diverging_palette(220, 20, s=80, l=70, sep=1, center="dark", as_cmap=True)
    # Make text larger
    plt.rcParams.update({'font.size': 30, 'font.family': 'Nimbus Sans'})

    for n in range(nPairs):
        fInd = n*2 + np.array([0, 1])
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
        ax.tick_params(axis='both', which='major', labelsize=22)
        ap.add_colorbar(ax=ax, ctgVal=ctgValSub, colorMap=cmap,
                         label='  Spd (m/s)', ticks=[-2, -1, 0, 1, 2])
        if savePlots:
            plt.savefig(fname=f'{plotTypeDirName}f{n*2+1}f{n*2+2}_' + \
                f'ellipses_noise{addRespNoise}_contrast_{ctr:.4f}.png',
                        bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()


    ###############
    # 2) GET MODEL ESTIMATES FOR DIFFERENT CONTRATS
    ###############

    plotTypeDirName = f'{plotDirName}2_estimates/'
    os.makedirs(plotTypeDirName, exist_ok=True)

    # Statistics interpolation parameters
    interpPoints = 11

    # Trim the edges that have border effects in estimation
    ctgTrim = 4
    inds2plot = torch.arange(ctgTrim, nCtg-ctgTrim)
    estimates = []
    ctgIndList = []
    statsList = []
    repeats = 5

    plt.rcParams.update({'font.size': 22, 'font.family': 'Nimbus Sans'})

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
        estTemp.append(ama.get_estimates(s=sPlt, method4est='MAP',
                                           addRespNoise=True).detach())
        ctgIndTemp.append(ctgIndTst)

    # Tidy estimates into tensors
    estimates = torch.tensor(torch.cat(estTemp), dtype=torch.float32)
    ctgIndRep = torch.cat(ctgIndTemp)
    # Get statistics of estimates
    statsList = au.get_estimate_statistics(estimates=estimates,
                                                ctgInd=ctgIndRep,
                                                quantiles=[0.16, 0.84])

    errorInterval = torch.cat((statsList['lowCI'].reshape(-1,1),
            statsList['highCI'].reshape(-1,1)), dim=-1).transpose(0,1)

    # Plot the estimates and CI
    fig, ax = plt.subplots()
    plot_estimate_statistics(ax=ax, estMeans=statsList['estimateMedian'][inds2plot],
        errorInterval=errorInterval[:,inds2plot], ctgVal=ctgVal[inds2plot], color='black')
    plt.xlabel('3D speed (m/s)')
    plt.ylabel('3D speed estimates (m/s)')
    # Set tick font size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Save the plot
    if savePlots:
        fig, ax = plt.gcf(), plt.gca()
        fig.tight_layout(rect=[0, 0, 0.95, 0.95])
        fig.set_size_inches(7, 6)
        plt.savefig(fname=f'{plotTypeDirName}model_estimates_contrast_{ctr:.4f}.png',
              bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

    # Get ctgInd in inds2plot
    ctgIndPlot = ctgIndRep[torch.isin(ctgIndRep, inds2plot)]
    estPlot = estimates[torch.isin(ctgIndRep, inds2plot)]
    # Make density plot
    jitter = torch.rand(len(ctgIndPlot)) * 0.05 - 0.025
    sns.scatterplot(x=ctgVal[ctgIndPlot]+jitter, y=estPlot,
                    color='black', alpha=0.1)
    sns.scatterplot(x=ctgVal[inds2plot],
                    y=statsList['estimateMedian'][inds2plot],
                    color='black', s=30) 
    # Set plot limits
    plt.ylim([ctgVal[inds2plot].min(), ctgVal[inds2plot].max()])
    plt.xlim([ctgVal[inds2plot].min(), ctgVal[inds2plot].max()])
    # Save the plot
    if savePlots:
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(8, 8)
        plt.savefig(fname=f'{plotTypeDirName}estimates_density_contrast_{ctr:.4f}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


    ###############
    # 3) GET MODEL ESTIMATES FOR DIFFERENT CONTRATS
    ###############

    plotTypeDirName = f'{plotDirName}3_posteriors/'
    os.makedirs(plotTypeDirName, exist_ok=True)

    # Compute posteriors
    posteriors = ama.get_posteriors(s=sPlt).detach()
    # Find the interpolated category indices and values
    ctg2plot = torch.tensor([9, 13, 17, 21, 23, 25])
    ctgValInterp = ama.ctgVal
    ctgIndInterp = find_interp_indices(ctgVal, ctgValInterp, ctgIndTst)
    ctg2plotInterp = (ctg2plot) * (interpPoints + 1)
    # Initialize figure with subplots
    nCols = 3
    nRows = 2
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(16,8))
    for i in range(len(ctg2plot)):
        inds = ctgIndInterp == ctg2plotInterp[i]
        postCtg = posteriors[inds,:]
        # Plot the posteriors
        ap.plot_posterior(ax=ax.flatten()[i], posteriors=postCtg,
                  ctgVal=ctgValInterp, trueVal=ctgVal[ctg2plot[i]])
        # Set axes title
        ax.flatten()[i].set_title(f'{ctgVal[ctg2plot[i]]:.1f} m/s', fontsize=20)
        # If it is the first row, remove x ticks
        if i < nCols:
            ax.flatten()[i].set_xticks([])
        else:
            ax.flatten()[i].set_xlabel('Speed (m/s)')
        # If it is first column, set y label
        if i % nCols == 0:
            ax.flatten()[i].set_ylabel('Posterior', fontsize=20)
        # Remove y ticks
        ax.flatten()[i].set_yticks([])
        ax.flatten()[i].set_ylim([0, 0.02])
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}posteriors_contrast{ctr:.4f}.png',
                  bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
    



# Initialize model
ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=10, respNoiseVar=respNoiseVar,
        pixelCov=pixelNoiseVar, ctgVal=ctgVal,
        samplesPerStim=samplesPerStim, nChannels=2)

ama.assign_filter_values(fNew=trainingDict['filters'])
ama.update_response_statistics()

# Remove covariance noise
ama.respCov = ama.respCovNoiseless.clone().detach()

# Interpolate class statistics
interpPoints = 11
ama.respCov = covariance_interpolation(covariance=ama.respCov.detach(),
                                       nPoints=interpPoints)
ama.respMean = mean_interpolation(mean=ama.respMean.detach(),
                                  nPoints=interpPoints)
ama.ctgVal = torch.tensor(linear_interpolation(y=ctgVal, nPoints=interpPoints),
                          dtype=torch.float32)

### plot how one stimulus's posterior changes with contrast
logContrasts = torch.linspace(start=0, end=-3, steps=200)
contrasts = 10**logContrasts


spd = 2
# Find the ctgInd corresponding to this speed
ctgIndSpd = torch.where(ctgVal == spd)[0]
# Get the index of a stimulus with this speed
stimInd = torch.where(ctgIndTst == ctgIndSpd)[0][0] + 11

# Multiply by contrast
stim = sTst[stimInd,:]
stimCtr = eo.repeat(stim, 'x -> (c) x', c=len(contrasts))
stimCtr = torch.einsum('ab,a->ab', stimCtr, contrasts)

# Get the posteriors
posteriors = ama.get_posteriors(s=stimCtr, addRespNoise=False).detach()
# Get the estimates
estimates = ama.get_estimates(s=stimCtr, addRespNoise=False).detach()

# Plot the estimates
fig, ax = plt.subplots()
plt.scatter(contrasts, estimates, color='black')
plt.xlabel('Multiplier')
plt.ylabel('3D speed estimate (m/s)')
plt.savefig(fname=f'{plotDirName}single_stim_contrast_estimate{stimInd}.png',
          bbox_inches='tight', pad_inches=0)
plt.close()

# Plot the posteriors with graded color
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=torch.min(logContrasts), vmax=torch.max(logContrasts))
colors = cmap(norm(logContrasts))
fig, ax = plt.subplots()
for i in range(len(contrasts)):
    ax.plot(ama.ctgVal, posteriors[i,:], color=colors[i])
plt.xlabel('Speed (m/s)')
plt.ylabel('Posterior')
# Add colormap
ap.add_colorbar(ax=ax, ctgVal=logContrasts, colorMap=cmap,
                 label='log-contrast', ticks=[-3, -2, -1, 0])
plt.savefig(fname=f'{plotDirName}single_stim_contrast_posterior{stimInd}.png',
          bbox_inches='tight', pad_inches=0)
plt.close()


