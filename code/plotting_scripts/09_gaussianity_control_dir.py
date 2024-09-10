##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm
import torch.distributions.multivariate_normal as mvn
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
plotDirName = f'./plots/gaussianity_test/dnK{dnK}_spd{spd}_noise{noise}_' + \
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
data = spio.loadmat('./data/ama_inputs/direction_looming/'
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Testing
dataTst = spio.loadmat('./data/ama_inputs/direction_looming/'
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
# 1) PLOT MODEL ESTIMATES AND CI
###############

plotTypeDirName = f'{plotDirName}1_estimates/'
os.makedirs(plotTypeDirName, exist_ok=True)

ci = [0.16, 0.84]  # Confidence interval
shiftX = True  # Shift the values to 0-360
repeats = 5  # Noise samples per stimuli

# Initialize new ama model
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                        ctgVal=ctgVal, samplesPerStim=samplesPerStim)
ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

# Get estimates for each stimulus and statistics
ll = ama.resp_2_ll(resp=respTstGauss)
post = ama.ll_2_posterior(ll=ll)
estimates = ama.posterior_2_estimate(posteriors=post,
                                        method4est='MAP').detach()
estStats = au.get_estimate_circular_statistics(estimates=estimates, ctgInd=ctgIndGauss,
                                                     quantiles=ci)
# Get estimates with shifted angles to match literature
estimatesShift = shift_angles(estimates)
estStatsShift = au.get_estimate_circular_statistics(estimates=estimatesShift,
                                                    ctgInd=ctgIndGauss, quantiles=ci)

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
    ctgIndPlot = invSortInd[ctgIndGauss]
else:
    lowCI = estStats[f'lowCI{stat}']
    highCI = estStats[f'highCI{stat}']
    estM = estStats[f'estimate{stat}']
    ctgValPlot = ctgVal
    estimatesPlot = estimates
    ctgIndPlot = ctgIndGauss
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
# 2) PLOT FRONTOPARALLEL VS BACK-FORTH FILTERS
#########################

plotTypeDirName = f'{plotDirName}2_filter_types/'
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

for k in range(len(filterIndList)):
    ### INITIALIZE AMA
    # Extract the name and indices of the filters
    tName = filterType[k]
    # Initialize ama model
    ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                           ctgVal=ctgVal, samplesPerStim=samplesPerStim)
    # Generate new AMA model with the indicated filters
    filterSubset = trainingDict['filters'][filterIndList[k],:]
    respSubset = respTstGauss[:,filterIndList[k]]
    # Assign to ama
    ama.assign_filter_values(fNew=filterSubset)
    ama.update_response_statistics()
    # Interpolate class statistics
    ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)

    ### OBTAIN THE ESTIMATES
    ll = ama.resp_2_ll(resp=respSubset)
    post = ama.ll_2_posterior(ll=ll)
    estimates = ama.posterior_2_estimate(posteriors=post,
                                            method4est='MAP').detach()

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
    bfSignAgree = (bfSign == bfCtg[ctgIndGauss])
    bfSignAgree = bfSignAgree.type(torch.float)
    # Get the left-right component
    leftRightComp = torch.sin(torch.deg2rad(estimates)) * 0.15
    # Convert to LR error
    lrError = torch.abs(leftRightComp - lrCtg[ctgIndGauss])
    # Get statistics on the components
    bfStats[tName] = au.get_estimate_statistics(estimates=bfSignAgree[notFpInds],
                                                ctgInd=ctgIndGauss[notFpInds])
    lrStats[tName] = au.get_estimate_statistics(estimates=lrError,
                                                ctgInd=ctgIndGauss)
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


