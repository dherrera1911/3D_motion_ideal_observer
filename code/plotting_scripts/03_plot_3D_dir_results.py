##########################################
# This script plots the main results for the 3D direction task.
# It plots:
# 1) Example stimuli
# 2) The learned filters
# 3) The filter response distributions
# 5) The model estimates scatter plot
# 6) Model performance (confidence intervals and towards-away accuracy)
#   with different filter subsets
# 7) The likelihood neuron tuning curves
# 
# Code author: Daniel Herrera-Esposito, dherrera1911 at gmail dot com
##########################################


##########################################
# This script plots the main results for the 3D direction task.
# It plots both example preprocessed stimuli, the filters
# learned by the model, and the performance of the model
# at 3D direction estimation with different filter subsets.
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
#### LOAD DATA
##############

# SPECIFY WHAT STIMULUS DATASET AND MODEL TO LOAD
savePlots = True
dnK = 2
spd = '0.15'
degStep = '7.5'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
loom = '1'
dspStd = '00'
plotDirName = f'./plots/3D_dir/dnK{dnK}_spd{spd}_noise{noise}_' + \
    f'degStep{degStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)

# SPECIFY THE INDICES OF DIFFERENT FILTER SUBSETS
allFiltersInd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
lrFiltersInd = np.array([0, 1, 2, 3])
bfFiltersInd = np.array([4, 5, 6, 7, 8, 9])
# Put different filters indices in a list
filterIndList = [allFiltersInd, lrFiltersInd, bfFiltersInd]
filterType = ['All', 'LR', 'BF']

# Specify interpolation and subsampling parameters
# Estimation parameters
interpPoints = 11 # Number of interpolation points between categories
samplesPerStim = 10 # Number of noise samples for stimulus initialization

# LOAD STIMULUS DATASET
# Training
data = spio.loadmat('./data/ama_inputs/'
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Testing
dataTst = spio.loadmat('./data/ama_inputs/'
  f'D3D-nStim_0300-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')

# PROCESS STIMULUS DATASET
nFrames = 15
# Keep only category angle of motion
ctgVal = ctgVal[1,:]
ctgValTst = ctgValTst[1,:]
# Get a vector of categories that is shifted, to match literature convention
ctgVal360 = shift_angles(ctgVal)
nCtg = len(ctgVal)
# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

##############
#### INITIALIZE TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Initialize ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                       ctgVal=ctgVal, samplesPerStim=samplesPerStim)

##############
#### PLOT MODEL OUTPUTS
##############

###############
# 0) PLOT NOISY NORMALIZED STIMULI
###############

plotTypeDirName = f'{plotDirName}0_noisy_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)

plotStimuli = True
nStimPlot = 2
if plotStimuli:
    for k in range(nCtg):
        # Step 1: filter rows by category
        indices = (ctgInd == k).nonzero(as_tuple=True)[0]
        s_k = s[indices]
        # random sample of rows
        sampleIndices = torch.randperm(s_k.shape[0])[:nStimPlot]
        sSample = s_k[sampleIndices]
        # Step 2: apply the function to generate noisy samples
        sNoisy = ama.preprocess(s=sSample)
        # Step 3: convert the 2D array sNoisy with shape (nStimPlot, df), into a 3D array
        sNoisy = unvectorize_1D_binocular_video(sNoisy, nFrames=nFrames)
        # Step 4: plot and save each matrix from the new sNoisy
        for i in range(sNoisy.shape[0]):
            plt.imshow(sNoisy[i,:,:].squeeze(), cmap='gray')
            ax = plt.gca()
            plt.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if savePlots:
                fileName = f'{plotTypeDirName}noisy_stim_spd{ctgVal[k]}_sample{i}.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()


###############
# 1) PLOT EACH FILTER
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
else:
    for k in range(nFilters):
        plt.subplot(1, nFilters, k+1)
        plt.imshow(fAll2D[k, :, :].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()


###############
# 2) PLOT RESPONSE ELLIPSES
###############

plotTypeDirName = f'{plotDirName}2_covariances/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Plot parameters
addRespNoise = False
colorCode = ['frontoparallel', 'depth']
colorLabel = ['Frontoparallel speed (m/s)', 'Towards-away speed (m/s)']

# Initialize ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                       ctgVal=ctgVal, samplesPerStim=samplesPerStim)
# Get the responses to the dataset
responses = ama.get_responses(s=s, addRespNoise=addRespNoise).detach()

# Subsample stimuli for better visualization
sSubs = 3
respSub = responses[::sSubs,:]
ctgIndSub = ctgInd[::sSubs]

# Subsample number of categories for better visualization
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

filterPairs = [[0,1], [5,6], [5,7], [5,8]]
filterPairs = [[0,1], [2,3], [4,5], [6,7], [8,9], [5,6], [5,7], [5,8]]

for c in range(len(colorCode)):
    cc = colorCode[c]
    if cc == 'frontoparallel':
        ctgValTrans = torch.sin(torch.deg2rad(ctgValSub)) * np.double(spd) 
    elif cc == 'depth':
        ctgValTrans = torch.cos(torch.deg2rad(ctgValSub)) * np.double(spd)
    elif cc == 'circular':
        ctgValTrans = ctgVal
    # Plot responses + ellipse for pair j,i
    #for i in range(nFilt):
    #    for j in range(i):
    for k in range(len(filterPairs)):
            j = filterPairs[k][0]
            i = filterPairs[k][1]
            fInd = np.array([j,i])
            pairCov = subsample_cov_inds(covariance=respCovSub, keepInds=fInd)
            fig, ax = plt.subplots(figsize=(7, 6.5))
            # Plot the responses
            ap.response_scatter(ax=ax, resp=respSub[:,fInd], ctgVal=ctgValTrans[ctgIndSub],
                                colorMap=cmap)
            ap.plot_ellipse_set(mean=respMeanSub[:,fInd], cov=pairCov,
                                ctgVal=ctgValTrans, colorMap=cmap, ax=ax)
            plt.xlabel(f'f{fInd[0]+1} response')
            plt.ylabel(f'f{fInd[1]+1} response')
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(axis='both', which='major', labelsize=24)
            ap.add_colorbar(ax=ax, ctgVal=ctgValTrans, colorMap=cmap,
                            label=f'{colorLabel[c]}', ticks=[-0.15, 0, 0.15],
                            orientation='horizontal')
            if savePlots:
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.set_size_inches(6.5, 7)
                plt.savefig(fname=f'{plotTypeDirName}code-{cc}_noise{addRespNoise}' +\
                      f'_f{j+1}f{i+1}.png', bbox_inches='tight', pad_inches=0.13)
                plt.close()
            else:
                plt.show()
    plt.hist(responses[ctgInd==12,0].detach().numpy(), bins=20, color='black')
    plt.xlabel(f'$f_{1}$ response')
    plt.ylabel('Count')
    plt.savefig(fname=f'{plotTypeDirName}code-{cc}_noise{addRespNoise}_f1_histogram.png',
          bbox_inches='tight', pad_inches=0.15)
    plt.close()


# PLOT RESPONSE ELLIPSES FOR PAIRS OF SPEEDS WITH SAME FRONTOPARALLEL SPEED
# Uses convention -180 to 180, 0 is towards observer
speedsList = [[0, -180], [15, 165], [30, 150], [45, 135], [60, 120], [75, 105]]

# Filter pair to plot
i=5
j=8

cmap = sns.diverging_palette(220, 20, s=80, l=70, sep=1, center="dark", as_cmap=True)
for si in range(len(speedsList)):
    speeds = speedsList[si]
    fpSpeed = torch.sin(torch.deg2rad(torch.tensor(speeds[0]))) * np.double(spd)
    # Remove classes not in this speed pair
    indRemove = torch.where(~torch.isin(ctgVal, torch.tensor(speeds)))[0]
    indKeep = torch.where(torch.isin(ctgVal, torch.tensor(speeds)))[0]
    respMeanSub = ama.respMean.detach()[indKeep,:]
    if addRespNoise:
        respCovSub = ama.respCov.detach()[indKeep,:,:]
    else:
        respCovSub = ama.respCovNoiseless.detach()[indKeep,:,:]
    cc = 'depth'
    ctgValTrans = torch.cos(torch.deg2rad(ctgVal[indKeep])) * np.double(spd)
    # Plot responses + ellipse for pair j,i
    fInd = np.array([j,i])
    pairCov = subsample_cov_inds(covariance=respCovSub, keepInds=fInd)
    fig, ax = plt.subplots()
    # Plot the responses
    colorLims = None # [-0.15, 0.15]
    ap.plot_ellipse_set(mean=respMeanSub[:,fInd], cov=pairCov, ctgVal=ctgValTrans,
                        colorLims=colorLims, colorMap=cmap, ax=ax)
    # Have no ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    # Set title of plot
    plt.title(f'Dirs [-180, 180]: {speeds[0]} {speeds[1]}', fontsize=12)
    if savePlots:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        #fig.set_size_inches(1.5, 1.5)
        fig.set_size_inches(2.5, 3)
        plt.savefig(fname=f'{plotTypeDirName}conditional_code-{cc}_noise{addRespNoise}' +\
            f'_f{j+1}f{i+1}_fpSpd_{fpSpeed:.4f}.png', bbox_inches='tight', pad_inches=0.13)
        plt.close()
    else:
        plt.show()


###############
# 3) PLOT MODEL ESTIMATES AND CI
###############

plotTypeDirName = f'{plotDirName}3_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

ci = [0.16, 0.84]  # Confidence interval
shiftX = True  # Shift the values to 0-360
repeats = 5  # Noise samples per stimuli

# Initialize new ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                        ctgVal=ctgVal, samplesPerStim=samplesPerStim)
ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

# Get estimates for each stimulus and statistics
estimates = []
ctgIndRep = ctgIndTst.repeat(repeats)
for n in range(repeats):
    estimates.append(ama.get_estimates(s=sTst, method4est='MAP'))
estimates = torch.cat(estimates)
estStats = au.get_estimate_circular_statistics(estimates=estimates, ctgInd=ctgIndRep,
                                                     quantiles=ci)
# Get estimates with shifted angles to match literature
estimatesShift = shift_angles(estimates)
estStatsShift = au.get_estimate_circular_statistics(estimates=estimatesShift,
                                                    ctgInd=ctgIndRep, quantiles=ci)

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
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_{stat}_shiftX{shiftX}.png',
          bbox_inches='tight', pad_inches=0.1)
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
    plt.savefig(fname=f'{plotTypeDirName}model_estimates_density_shift{shiftX}2.png',
          bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


#########################
# 4) PLOT FRONTOPARALLEL VS BACK-FORTH FILTERS
#########################

plotTypeDirName = f'{plotDirName}4_filter_types/'
os.makedirs(plotTypeDirName, exist_ok=True)

quantiles = [0.16, 0.84]
method4est = 'MAP'
repeats = 5
addRespNoise = True
statsNoise = True

ctgValLR = torch.sin(torch.deg2rad(ctgVal)) * 0.15
# Round to 4 decimals
ctgValLR = torch.round(ctgValLR * 10000) / 10000
# Make text larger
plt.rcParams.update({'font.size': 30, 'font.family': 'Nimbus Sans'})

lrStats = {}
bfStats ={}

for ft in range(len(filterIndList)):
    ### INITIALIZE AMA
    # Extract the name and indices of the filters
    tName = filterType[ft]
    # Initialize ama model
    ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                           ctgVal=ctgVal, samplesPerStim=samplesPerStim)
    # Generate new AMA model with the indicated filters
    filterSubset = trainingDict['filters'][filterIndList[ft],:]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

    ### OBTAIN THE ESTIMATES
    estimates = []
    ctgIndList = []
    # Loop over repeats
    for r in range(repeats):
        print('Repeat: ', r)
        estimates.append(ama.get_estimates(
          s=sTst, method4est=method4est, addRespNoise=addRespNoise).detach())
        ctgIndList.append(ctgIndTst)
    estimates = torch.cat(estimates)
    ctgIndRep = torch.cat(ctgIndList)

    ### PROCESS ESTIMATES, AND COLLAPSE CATEGORIES WITH SAME LR COMPONENT
    # Get LR and BF components of categories
    bfCtg = torch.sign(torch.cos(torch.deg2rad(ctgVal)))
    bfCtg[torch.abs(ctgVal)==90] = 0
    lrCtg = torch.sin(torch.deg2rad(ctgVal)) * 0.15
    # Get towards-away sign of the estimates
    bfSign = torch.sign(torch.cos(torch.deg2rad(estimates)))
    # Find estimates that are frontoparallel plane, to remove from bf statistics
    tol = 0
    absEstimate = torch.abs(estimates)
    # get non-frontoparallel estimates indices to remove later
    notFpInds = torch.where((absEstimate > 90+tol) | (absEstimate < 90-tol))
    # Get whether the estimate has the correct bf sign
    bfSignAgree = (bfSign == bfCtg[ctgIndRep])
    bfSignAgree = bfSignAgree.type(torch.float)
    # Get the left-right component
    leftRightComp = torch.sin(torch.deg2rad(estimates)) * 0.15
    # Convert to LR error
    lrError = torch.abs(leftRightComp - lrCtg[ctgIndRep])
    # Get statistics on the components
    bfStats[tName] = au.get_estimate_statistics(estimates=bfSignAgree[notFpInds],
                                                ctgInd=ctgIndRep[notFpInds])
    lrStats[tName] = au.get_estimate_statistics(estimates=lrError,
                                                ctgInd=ctgIndRep)
    # Group categories by LR speed
    groupedCtg = torch.unique(ctgValLR)
    nGrCtg = len(groupedCtg)
    bfStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
    lrStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
    for n in range(nGrCtg):
        inds = torch.where(ctgValLR == groupedCtg[n])[0]
        bfStats[tName]['meanGrouped'][n] = torch.mean(bfStats[tName]['estimateMean'][inds])
        lrStats[tName]['meanGrouped'][n] = torch.mean(lrStats[tName]['estimateMean'][inds])


### PLOT PERFORMANCE IN BACK FORTH AND LEFT RIGHT ESTIMATION AS FUNCTION OF LR SPEED

plt.rcParams.update({'font.size': 18, 'font.family': 'Nimbus Sans'})
ms = 7 # Marker size
lw = 2 # Line width
fs = 18 # Font size

# Plot back-forth error
nGrCtg = len(groupedCtg)
plotInds = torch.arange(1, nGrCtg-1)

plt .plot(groupedCtg[plotInds], 1-bfStats['All']['meanGrouped'][plotInds],
          'black', marker='o', markersize=ms , linewidth=lw, label='All')
plt.plot(groupedCtg[plotInds], 1-bfStats['LR']['meanGrouped'][plotInds],
         'tab:orange', marker='o', markersize=ms, linewidth=lw, label='Frontoparallel')
plt.plot(groupedCtg[plotInds], 1-bfStats['BF']['meanGrouped'][plotInds],
         'tab:purple', marker='o', markersize=ms, linewidth=lw, label='Towards-away')

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
xticks = [-0.15, 0, 0.15]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel('Frontoparallel speed (m/s)', fontsize=fs)
plt.ylabel('Towards-away confusions', fontsize=fs)
plt.legend(loc='upper center', fontsize=16)
if savePlots:
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    plt.savefig(fname=f'{plotTypeDirName}towards_away_conf_respNoise_{addRespNoise}_'
                f'_noisyCov_{statsNoise}.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

# Plot left-right estimates
plt.plot(groupedCtg, lrStats['All']['meanGrouped'], 'black',
         marker='o', markersize=ms, linewidth=lw, label='All')
plt.plot(groupedCtg, lrStats['LR']['meanGrouped'], 'tab:orange',
         marker='o', markersize=ms , linewidth=lw, label='FP')
plt.plot(groupedCtg, lrStats['BF']['meanGrouped'], 'tab:purple',
         marker='o', markersize=ms, linewidth=lw, label='TA')
plt.ylim(0, 0.06)
plt.yticks([0, 0.025, 0.05])
xticks = [-0.15, 0, 0.15]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=fs)
plt.xlabel('Frontoparallel speed (m/s)', fontsize=fs)
plt.ylabel('Frontoparallel MAE (m/s)', fontsize=fs)
if savePlots:
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    plt.savefig(fname=f'{plotTypeDirName}bf_respNoise_{addRespNoise}_'
                f'_noisyCov_{statsNoise}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()


###############
# 5) PLOT POSTERIORS
###############

plotTypeDirName = f'{plotDirName}5_posteriors/'
os.makedirs(plotTypeDirName, exist_ok=True)
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

# Initialize ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                       ctgVal=ctgVal, samplesPerStim=samplesPerStim)
# Interpolate class statistics
ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

# Compute posteriors
repeats = 5
posteriors = []
ctgIndRep = []
for r in range(repeats):
    print('Repeat: ', r)
    posteriors.append(ama.get_posteriors(s=sTst).detach())
    ctgIndRep.append(ctgIndTst)
posteriors = torch.cat(posteriors)
posteriors = posteriors.type(torch.float32)
ctgIndRep = torch.cat(ctgIndRep)

# Find the interpolated category indices and values
ctgValInterp = ama.ctgVal
ctgIndInterp = find_interp_indices(ctgVal, ctgValInterp, ctgIndRep)
ctg2plot = torch.arange(0, 25)
ctg2plotInterp = (ctg2plot) * (interpPoints + 1)

# Plot the likelihood neurons
for i in range(len(ctg2plot)):
    fig, ax = plt.subplots(figsize=(2.2,1.5))
    # Class posteriors
    posteriorCtg = posteriors[:, ctg2plotInterp[i]]
    # Plot response of likelihood neurons
    ap.plot_posterior_neuron(ax=ax, posteriorCtg=posteriorCtg,
                             ctgInd=ctgIndRep, ctgVal=ctgVal360,
                             trueVal=ctgVal360[ctg2plot[i]])
    # If it is the first row, remove x ticks
    ax.set_xlabel('Direction (deg)')
    # If it is first column, set y label
    ax.set_ylabel('Likelihood neuron response', fontsize=8)
    # Remove y ticks
    ax.set_yticks([])
    # Set vertical bars showing change of directions
    ax.axvline(90-1.86, color='grey', alpha=0.5)
    ax.axvline(90+1.86, color='grey', alpha=0.5)
    ax.axvline(270-1.86, color='grey', alpha=0.5)
    ax.axvline(270+1.86, color='grey', alpha=0.5)
#    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title(f'Selectivity: {ctgVal360[ctg2plot[i]]:.2f} deg', fontsize=10)
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}2_likelihood_neuron_dir_{ctgVal[ctg2plot[i]]:.2f}.png',
              bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

