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

# Process stimulus dataset
nStim = s.shape[0]
df = s.shape[1]
nFrames = 15
nCtg = len(ctgVal)
# Get only angle of motion
ctgVal = ctgVal[1,:]
ctgValTst = ctgValTst[1,:]
# Get a vector of categories that is shifted, to match literature convention
ctgVal360 = shift_angles(ctgVal)
# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

# Put different filters indices in a list
filterInds = [allFiltersInd, lrFiltersInd, bfFiltersInd]
filterType = ['All', 'LR', 'BF']


##############
#### INITIALIZE TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Initialize random AMA model
respNoiseVar = trainingDict['respNoiseVar']
pixelNoiseVar = trainingDict['pixelNoiseVar']

ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=10, respNoiseVar=respNoiseVar,
        pixelCov=pixelNoiseVar, ctgVal=ctgVal,
        samplesPerStim=samplesPerStim, nChannels=2)

# Assign the learned filters to the model
ama.assign_filter_values(fNew=trainingDict['filters'])
ama.update_response_statistics()

##############
#### PLOT MODEL OUTPUTS
##############

###############
# 0) PLOT NOISY NORMALIZED STIMULI
###############

plotTypeDirName = f'{plotDirName}0_noisy_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)

nStimPlot = 2
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

responses = ama.get_responses(s=s, addRespNoise=addRespNoise).detach()
if addRespNoise:
    respCov = ama.respCov.clone().detach()
else:
    respCov = ama.respCovNoiseless.clone().detach()

# Subsample stimuli for better visualization
sSubs = 3
respSub = responses[::sSubs,:]
ctgIndSub = ctgInd[::sSubs]

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

filterPairs = [[0,1], [5,6], [5,7], [5,8]]

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
    plt.hist(responses[ctgInd==12,0].detach().numpy(), bins=20)
    plt.xlabel(f'$f_{1}$ response')
    plt.ylabel('Count')
    plt.savefig(fname=f'{plotTypeDirName}code-{cc}_noise{addRespNoise}_f1_histogram.png',
          bbox_inches='tight', pad_inches=0.15)
    plt.close()


# PLOT RESPONSE ELLIPSES FOR PAIRS OF SPEEDS WITH SAME FRONTOPARALLEL SPEED
# Uses convention -180 to 180, 0 is towards observer
speedsList = [[0, -180], [15, 165], [30, 150], [45, 135], [60, 120], [75, 105]]

for si in range(len(speedsList)):
    speeds = speedsList[si]
    fpSpeed = torch.sin(torch.deg2rad(torch.tensor(speeds[0]))) * np.double(spd)
    # Find indices of classes different than speeds
    indRemove = torch.where(~torch.isin(ctgVal, torch.tensor(speeds)))[0]
    indKeep = torch.where(torch.isin(ctgVal, torch.tensor(speeds)))[0]
    ctgValSub, ctgIndSub, respSub = remove_categories(removeCtg=indRemove,
            ctgVal=ctgVal, ctgInd=ctgInd, s=responses)
    respMeanSub = ama.respMean.detach()[indKeep,:]
    if addRespNoise:
        respCovSub = ama.respCov.detach()[indKeep,:,:]
    else:
        respCovSub = ama.respCovNoiseless.detach()[indKeep,:,:]
    cc = 'depth'
    ctgValTrans = torch.cos(torch.deg2rad(ctgValSub)) * np.double(spd)
    # Plot responses + ellipse for pair j,i
    # Filter pair to plot
    i=8
    j=5
    fInd = np.array([j,i])
    pairCov = subsample_cov_inds(covariance=respCovSub, keepInds=fInd)
    fig, ax = plt.subplots()
    # Plot the responses
    colorLims = None # [-0.15, 0.15]
#    ap.response_scatter(ax=ax, resp=respSub[:,fInd], ctgVal=ctgValTrans[ctgIndSub],
#                        colorLims=[-0.15, 0.15], colorMap=cmap)
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
# INTERPOLATE AMA CLASSES
###############
# Interpolate class statistics
ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)
# Clone interpolated statistics to re-assign later
respMeanInt = ama.respMean.clone().detach()
respCovInt = ama.respCov.clone().detach()


###############
# 3) PLOT MODEL ESTIMATES AND CI
###############

plotTypeDirName = f'{plotDirName}3_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

ci = [0.16, 0.84]  # Confidence interval
shiftX = True  # Shift the values to 0-360

# Get model estimates
repeats = 5  # Noise samples per stimuli
estimates = []
ctgIndRep = ctgIndTst.repeat(repeats)
for n in range(repeats):
    estimates.append(ama.get_estimates(s=sTst, method4est='MAP'))
estimates = torch.cat(estimates) 
estStats = au.get_estimate_circular_statistics(estimates=estimates, ctgInd=ctgIndRep,
                                                     quantiles=ci)
# Get shifted estimates
estimatesShift = shift_angles(estimates)
estStatsShift = au.get_estimate_circular_statistics(estimates=estimatesShift, ctgInd=ctgIndRep,
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
for ft in range(len(filterType)):
    # Extract the name and indices of the filters
    tName = filterType[ft]
    fInds = filterInds[ft]
    # Generate new AMA model with the indicated filters
    ama.assign_filter_values(fNew=trainingDict['filters'][fInds,:])
    ama.respMean = respMeanInt[:,fInds]
    ama.respCov = subsample_cov_inds(covariance=respCovInt, keepInds=fInds)
    ### NEED TO IMPLEMENT STATSNOISE=FALSE
    # Obtain the estimates
    estimates = []
    ctgIndList = []
    # Loop over repeats
    for r in range(repeats):
        print('Repeat: ', r)
        estimates.append(ama.get_estimates(
          s=sTst, method4est=method4est, addRespNoise=addRespNoise).detach())
        ctgIndList.append(ctgIndTst)
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
    lrStats[tName] = au.get_estimate_statistics(estimates=lrError,
                                                ctgInd=ctgIndReps)
    bfStats[tName] = au.get_estimate_statistics(estimates=bfSignAgree,
                                                ctgInd=ctgIndReps)
    # Group categories by LR speed
    groupedCtg = torch.unique(ctgValLR)
    nGrCtg = len(groupedCtg)
    bfStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
    lrStats[tName]['meanGrouped'] = torch.zeros(nGrCtg)
    for n in range(nGrCtg):
        inds = torch.where(ctgValLR == groupedCtg[n])[0]
        bfStats[tName]['meanGrouped'][n] = torch.mean(bfStats[tName]['estimateMean'][inds])
        lrStats[tName]['meanGrouped'][n] = torch.mean(lrStats[tName]['estimateMean'][inds])

# Plot back-forth error
plt .plot(groupedCtg, 1-bfStats['All']['meanGrouped'], 'black',
          marker='o', markersize=12 , linewidth=4, label='All')
plt.plot(groupedCtg, 1-bfStats['LR']['meanGrouped'], 'tab:orange',
         marker='o', markersize=12, linewidth=4, label='Frontoparallel')
plt.plot(groupedCtg, 1-bfStats['BF']['meanGrouped'], 'tab:purple',
         marker='o', markersize=12, linewidth=4, label='Towards-away')
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
xticks = [-0.15, 0, 0.15]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=26)
plt.xlabel('Frontoparallel speed (m/s)', fontsize=30)
plt.ylabel('Towards-away confusions', fontsize=30)
plt.legend(loc='upper center', fontsize=26)
if savePlots:
    fig = plt.gcf()
    fig.set_size_inches(8.5, 7)
    plt.savefig(fname=f'{plotTypeDirName}towards_away_conf_respNoise_{addRespNoise}_'
                f'_noisyCov_{statsNoise}.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

# Plot left-right estimates
plt.plot(groupedCtg, lrStats['All']['meanGrouped'], 'black',
         marker='o', markersize=12, linewidth=4, label='All')
plt.plot(groupedCtg, lrStats['LR']['meanGrouped'], 'tab:orange',
         marker='o', markersize=12 , linewidth=4, label='FP')
plt.plot(groupedCtg, lrStats['BF']['meanGrouped'], 'tab:purple',
         marker='o', markersize=12, linewidth=4, label='TA')
plt.ylim(0, 0.06)
plt.yticks([0, 0.025, 0.05])
xticks = [-0.15, 0, 0.15]
plt.xticks(xticks, [str(xtick) for xtick in xticks])
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=26)
plt.xlabel('Frontoparallel speed (m/s)', fontsize=30)
plt.ylabel('Frontoparallel MAE (m/s)', fontsize=30)
if savePlots:
    fig = plt.gcf()
    fig.set_size_inches(8.8, 7)
    plt.savefig(fname=f'{plotTypeDirName}bf_respNoise_{addRespNoise}_'
                f'_noisyCov_{statsNoise}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
else:
    plt.show()



###############
# 4) PLOT COVARIANCE FUNCTIONS
###############

plotTypeDirName = f'{plotDirName}4_covariance_plots/'
os.makedirs(plotTypeDirName, exist_ok=True)

ap.plot_covariance_values(covariances=[ama.respCov.detach()],
                          xVal=[ctgVal], sizeList=[3], showPlot=False)
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}covariances_noisy.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

ap.plot_covariance_values(covariances=[ama.respCovNoiseless.detach()],
                          xVal=[ctgVal], sizeList=[3], showPlot=False)
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}covariances_noiseless.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()


ap.plot_covariance_values(covariances=[ama.respCov.detach()],
                          xVal=[ctgValPlot], sizeList=[4], showPlot=False)
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}covariances_litX.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()


ap.plot_covariance_values(covariances=[ama.respCovNoiseless.detach()],
                          xVal=[ctgValPlot], sizeList=[4], showPlot=False)
if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}covariances_noiseless_litX.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()


###############
# 5) PLOT POSTERIORS
###############

plotTypeDirName = f'{plotDirName}5_posteriors/'
os.makedirs(plotTypeDirName, exist_ok=True)

# Interpolate covariances
interpPoints = 11
ama, trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))
ama.interpolate_class_statistics(nPoints=interpPoints, method='spline',
                                 variableType='circular')

# Compute posteriors
posteriors = ama.get_posteriors(s=sTst).detach()
posteriors = posteriors.type(torch.float32)

# Find the interpolated category indices and values
ctgValInterp = ama.ctgVal
ctgIndInterp = au.find_interp_indices(ctgVal, ctgValInterp, ctgIndTst)
#ctg2plot = torch.tensor([12, 15, 18, 21, 24])
ctg2plot = torch.tensor([12, 14, 16, 18, 20, 22, 24])
ctg2plotInterp = (ctg2plot) * (interpPoints + 1)

# Plot posterior distribution for each class
ap.plot_posteriors(posteriors=posteriors, ctgInd=ctgIndInterp,
                   ctgVal=ctgValInterp, ctg2plot=ctg2plotInterp,
                   traces2plot=100, showPlot=False, maxColumns=len(ctg2plot))

# Get figure and axes
fig, ax = plt.gcf(), plt.gca()
# Get first axes of figure
ax = fig.axes[0]
ax.set_ylabel('Posterior probability')
for a in range(len(fig.axes)):
    # Remove y ticks
    if a != 0:
        fig.axes[a].yaxis.set_visible(False)
    # Set axes title
    fig.axes[a].set_title(f'{ctgVal[ctg2plot[a]]} deg')
# Set figure size
fig.set_size_inches(13, 3)

if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}posteriors.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

# Plot response of likelihood neurons

ap.plot_posterior_neuron(posteriors=posteriors, ctgInd=ctgIndInterp,
                         ctg2plot=ctg2plotInterp, ctgVal=ctgValInterp,
                         showPlot=False, maxColumns=len(ctg2plot))

# Get figure and axes
fig, ax = plt.gcf(), plt.gca()
# Get first axes of figure
ax = fig.axes[0]
ax.set_ylabel('Posterior probability')
for a in range(len(fig.axes)):
    # Remove y ticks
    if a != 0:
        fig.axes[a].yaxis.set_visible(False)
    # Set axes title
    fig.axes[a].set_title(f'{ctgVal[ctg2plot[a]]} deg')
# Set figure size
fig.set_size_inches(13, 3)

if savePlots:
    plt.savefig(fname=f'{plotTypeDirName}likelihood_neurons.png',
          bbox_inches='tight', pad_inches=0)
    plt.close()
else:
    plt.show()

