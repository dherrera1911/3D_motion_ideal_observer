##########################################
# This script plots the 3D direction estimation model's outputs for
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

savePlots = True
dnK = 2
spd = '0.15'
degStep = '7.5'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00'
plotDirName = f'./plots/3D_dir_contrast/dnK{dnK}_spd{spd}_noise{noise}_' + \
    f'degStep{degStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# Set the indices of which are the monocular and the binocular filters
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
lrFiltersInd = np.array([0, 1, 2, 3, 4])
bfFiltersInd = np.array([5, 6, 7, 8, 9])
filterInds = [allFiltersInd, lrFiltersInd, bfFiltersInd]
filterType = ['All', 'LR', 'BF']

##############
#### LOAD STIMULI
##############
# TRAINING
data = spio.loadmat('./data/ama_inputs/'
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# TESTING
dataTst = spio.loadmat('./data/ama_inputs/'
  f'D3D-nStim_0300-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')

# Extract some properties of the dataset
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
# Get only angle of motion
ctgVal = ctgVal[1,:]
ctgValTst = ctgValTst[1,:]
nCtg = len(ctgVal)

# Also get a vector of categories that is shifted, to match
# literature convention
ctgVal360 = shift_angles(ctgVal)

# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

# <codecell>
##############
#### LOAD TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'

trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

##############
#### PLOT MODEL OUTPUTS FOR DIFFERENT CONTRASTS
##############

contrasts = [1, 0.5, 0.25, 0.125, 0.0625, 0.0312]

for c in range(len(contrasts)):

    # Initialize random AMA model
    samplesPerStim = 10
    respNoiseVar = trainingDict['respNoiseVar']
    pixelNoiseVar = trainingDict['pixelNoiseVar']
    ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=10, respNoiseVar=respNoiseVar,
            pixelCov=pixelNoiseVar, ctgVal=ctgVal,
            samplesPerStim=samplesPerStim, nChannels=2)
    # Put trained filters into the model
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

    # Plot parameters
    addRespNoise = False
    colorCode = ['frontoparallel', 'depth']
    colorLabel = ['Frontoparallel speed (m/s)', 'Towards-away speed (m/s)']

    nFilt = 10

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
    ctgSubs = 2
    indKeep = np.arange(nCtg)[np.arange(nCtg) % ctgSubs == 0]
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
    plt.rcParams.update({'font.size': 32, 'font.family': 'Nimbus Sans'})

    filterPairs = [[0,1], [6,8]]

    for k in range(len(colorCode)):
        cc = colorCode[k]
        if cc == 'frontoparallel':
            ctgValTrans = torch.sin(torch.deg2rad(ctgValSub)) * np.double(spd) 
        elif cc == 'depth':
            ctgValTrans = torch.cos(torch.deg2rad(ctgValSub)) * np.double(spd)
        elif cc == 'circular':
            ctgValTrans = ctgVal
        # Plot responses + ellipse for pair j,i
        for i in range(len(filterPairs)):
            fInd = np.array(filterPairs[i])
            pairCov = subsample_cov_inds(covariance=respCovSub, keepInds=fInd)
            fig, ax = plt.subplots(figsize=(7, 6.5))
            # Plot the responses
            ap.response_scatter(ax=ax, resp=respSub[:,fInd],
                                ctgVal=ctgValTrans[ctgIndSub], colorMap=cmap)
            ap.plot_ellipse_set(mean=respMeanSub[:,fInd], cov=pairCov,
                                ctgVal=ctgValTrans, colorMap=cmap, ax=ax)
            plt.xlabel(f'f{fInd[0]+1} response')
            plt.ylabel(f'f{fInd[1]+1} response')
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(axis='both', which='major', labelsize=24)
            ap.add_colorbar(ax=ax, ctgVal=ctgValTrans, colorMap=cmap,
                            label=f'{colorLabel[k]}', ticks=[-0.15, 0, 0.15],
                            orientation='horizontal')
            if savePlots:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.set_size_inches(6.5, 7)
                plt.savefig(fname=f'{plotTypeDirName}code-{cc}_noise{addRespNoise}' +\
                    f'_f{fInd[0]+1}f{fInd[1]+1}_contrast_{ctr:.4f}.png',
                            bbox_inches='tight', pad_inches=0.13)
                plt.close()
            else:
                plt.show()

    ###############
    # 2) PLOT MODEL ESTIMATES AND CI
    ###############

    plotTypeDirName = f'{plotDirName}2_estimates/'
    os.makedirs(plotTypeDirName, exist_ok=True)

    # Estimation parameters
    interpPoints = 11

    ci = [0.16, 0.84]  # Confidence interval
    shiftX = True  # Shift the values to 0-360

    # Interpolate class statistics
    ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

    # Get model estimates
    repeats = 5  # Noise samples per stimuli
    estimates = []
    ctgIndRep = ctgIndTst.repeat(repeats)
    for n in range(repeats):
        estimates.append(ama.get_estimates(s=sPlt, method4est='MAP'))
    estimates = torch.cat(estimates) 
    estStats = au.get_estimate_circular_statistics(estimates=estimates, ctgInd=ctgIndRep,
                                                         quantiles=ci)
    # Get shifted estimates
    estimatesShift = shift_angles(estimates)
    estStatsShift = au.get_estimate_circular_statistics(estimates=estimatesShift,
                                                        ctgInd=ctgIndRep,
                                                         quantiles=ci)

    # Wrap the estimates stats to be the closest to ctgVal (e.g. if estimate is 5
    # for the category 355, then wrap to 365)
    estStats = match_circular_stats(ctgVal=ctgVal, estimateStats=estStats)
    estStatsShift = match_circular_stats(ctgVal=ctgVal360, estimateStats=estStatsShift)

    # Select statistics and angle convention to plot
    stat = 'Median'
    if shiftX:
        ctgValPlot = ctgVal360
        lowCI = estStatsShift[f'lowCI{stat}']
        highCI = estStatsShift[f'highCI{stat}']
        estM = estStatsShift[f'estimate{stat}']
        # Sort ctgVal and estimates to be in increasing order
        ctgValPlot, sortInd = torch.sort(ctgValPlot)
        lowCI = lowCI[sortInd]
        highCI = highCI[sortInd]
        estM = estM[sortInd]
        estimatesPlot = estimatesShift
        # Each element in ctgInd change its value to the sorted index
        # Create inverse mapping
        invSortInd = torch.argsort(sortInd)
        ctgIndPlot = invSortInd[ctgIndRep]
    else:
        lowCI = estStats[f'lowCI{stat}']
        highCI = estStats[f'highCI{stat}']
        estM = estStats[f'estimate{stat}']
        ctgValPlot = ctgVal
        estimatesPlot = estimates
        ctgIndPlot = ctgIndRep
    errorInterval = torch.cat((lowCI.reshape(-1,1), highCI.reshape(-1,1)), dim=-1).transpose(0,1)

    # Add first point to the end, to close the loop
    ctgValPlot = torch.cat((ctgValPlot, ctgValPlot[0].unsqueeze(0)+360))
    estM = torch.cat((estM, estM[0].unsqueeze(0)+360))
    errorInterval = torch.cat((errorInterval, errorInterval[:,0].unsqueeze(1)+360), dim=1)

    # Plot the estimates and CI
    plt.rcParams.update({'font.size': 32, 'font.family': 'Nimbus Sans'})
    fig, ax = plt.subplots()
    plot_estimate_statistics(ax=ax, estMeans=estM, errorInterval=errorInterval,
        ctgVal=ctgValPlot, color='black')
    ax.set_xlabel('3D direction (deg)')
    ax.set_ylabel('3D direction estimate (deg)')
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.tick_params(axis='both', which='major', labelsize=26)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(fname=f'{plotTypeDirName}model_estimates_{stat}_shiftX{shiftX}_' + \
            f'contrast_{ctr:.4f}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

    ### Plot estimate density
    # Duplicate the estimates of the first category to the end, to close the loop
    estimatesPlot = torch.cat((estimatesPlot, estimatesPlot[ctgIndPlot==0]))
    nEstimates = torch.sum(ctgIndPlot==0)
    # Repeat ctgIndPlot nEstimates times
    ctgIndPlot = torch.cat((ctgIndPlot, torch.full((nEstimates,), nCtg)))
    # Subsample the estimates for better visualization
    subs = 5
    estimatesSubs = estimatesPlot[::subs]
    ctgIndSub = ctgIndPlot[::subs]

    # scatter points without line
    jitter = (np.random.random(size=len(ctgIndSub))-0.5) * 5
    plt.scatter(x=ctgValPlot[ctgIndSub]+jitter, y=estimatesSubs, color="black",
                alpha=0.15, s=25, linewidth=0)
    # Identity line
    #plt.plot(np.sort(ctgValPlot), np.sort(ctgValPlot), color='black', linestyle='--',
    #         linewidth=5)
    plt.xlim(0,360)
    plt.ylim(0,360)
    plt.xticks([0,90,180,270,360])
    plt.yticks([0,90,180,270,360])
    plt.xlabel('3D speed (m/s)')
    plt.ylabel('3D speed estimates (m/s)')
    # Set tick font size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=26)

    # Setting xticks to category values for clarity
    plt.xlabel('3D direction (deg)', fontsize=28)
    plt.ylabel('3D direction estimate (deg)', fontsize=30)
    if savePlots:
        fig = plt.gcf()
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(fname=f'{plotTypeDirName}model_estimates_density_shift{shiftX}' + \
            f'contrast_{ctr:.4f}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()



