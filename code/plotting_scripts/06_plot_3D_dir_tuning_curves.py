#########################################
# This script plots the tuning curves of the 3D direction
# likelihood neurons in response to stimuli similar to
# those used in Czuba et al. where the 4 quadrants with
# each combination of posiitve/negative retinal speed for each eye
# have the same size. This is equivalent to having
# the target be very close to the eyes.
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
#spd = '0.15'
spd = '0.0024'
degStep = '7.5'
noise = '0.0100' # 0.0014, 0.0035, 0.0070, 0.0084, 0.0105, 0.0123, 0.0175, 0.0350
#loom = '1'
loom = '0'
dspStd = '00'
plotDirName = f'./plots/3D_dir/dnK{dnK}_spd{spd}_noise{noise}_' + \
    f'degStep{degStep}_loom{loom}/'
os.makedirs(plotDirName, exist_ok=True)
# Specify number of interpolation points
interpPoints = 11 # Number of interpolation points between categories

# LOAD DATASET USED TO TRAIN THE MODEL
#train_stim_dir = 'direction_looming/'
train_stim_dir = 'direction_close_target_tuning/'
data = spio.loadmat('./data/ama_inputs/' + train_stim_dir +
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
s = contrast_stim(s=s, nChannels=2)
# Keep only angle of motion as class
ctgVal = ctgVal[1,:]
nCtg = len(ctgVal)


##############
#### INITIALIZE TRAINED MODEL
##############

modelFile = f'./data/trained_models/' \
    f'ama_3D_dir_empirical_dnK_{dnK}_spd_{spd}_' \
    f'degStep_{degStep}_noise_{noise}_loom_{loom}_dspStd_{dspStd}_dict.pt'
trainingDict = torch.load(modelFile, map_location=torch.device('cpu'))

# Initialize ama model
samplesPerStim = 10 # Number of noise samples for stimulus initialization
ama = init_trained_ama(amaDict=trainingDict, sAll=s, ctgInd=ctgInd,
                       ctgVal=ctgVal, samplesPerStim=samplesPerStim)
# Interpolate class statistics
ama = interpolate_circular_ama(ama=ama, interpPoints=interpPoints)


##############
#### LOAD DATA TO DO TUNING CURVES
##############

# Dataset with speeds like Czuba et al.
#spd = 0.0024
spd = '0.15'
test_stim_dir = 'direction_looming/'
#loom = 0
loom = '1'
dataTst = spio.loadmat('./data/ama_inputs/' + test_stim_dir +
  f'D3D-nStim_0300-spd_{spd}-degStep_{degStep}-'
  f'dspStd_{dspStd}-dnK_{dnK}-loom_{loom}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
sTst = contrast_stim(s=sTst, nChannels=2)
nFrames = 15


##############
#### Export some stimuli
##############

plotTypeDirName = f'{plotDirName}6bis_literature_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)

plotStimuli = False
nStimPlot = 4
if plotStimuli:
    for k in range(nCtg):
        # LITERATURE STIM
        # Step 1: filter rows by category
        indices = (ctgIndTst == k).nonzero(as_tuple=True)[0]
        s_k = sTst[indices]
        # random sample of rows
        sSample = s_k[:nStimPlot]
        # Step 2: apply the function to generate noisy samples
        #sNoisy = ama.preprocess(s=sSample)
        sNoisy = sSample
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
                fileName = f'{plotTypeDirName}close_stim_dir{ctgVal[k]}_sample{i}.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()

        # FAR STIM
        # Step 1: filter rows by category
        indices = (ctgInd == k).nonzero(as_tuple=True)[0]
        s_k = s[indices]
        # random sample of rows
        sSample = s_k[:nStimPlot]
        # Step 2: apply the function to generate noisy samples
        sNoisy = sSample
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
                fileName = f'{plotTypeDirName}far_stim_dir{ctgVal[k]}_sample{i}.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()


##############
#### EXPORT FILTERS
##############

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
# PLOT LIKELIHOOD NEURON TUNING CURVE FOR TEST STIMULI
###############

plotTypeDirName = f'{plotDirName}6_posteriors_literature_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

# Make a category vector with literature angle convention
ctgVal360 = shift_angles(ctgVal)

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

# Find the category indices of the stimuli in the interpolated values
ctgValInterp = ama.ctgVal
ctgIndInterp = find_interp_indices(ctgVal, ctgValInterp, ctgIndRep)

# Likelihood neuron selectivities to plot
#ctg2plot = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
ctg2plot = torch.arange(25)
ctg2plotInterp = (ctg2plot) * (interpPoints + 1)

quantiles = [0.40, 0.60]
# Plot the likelihood neurons
for i in range(len(ctg2plot)):
    fig, ax = plt.subplots(figsize=(2.2,1.5))
    # Class posteriors
    posteriorCtg = posteriors[:, ctg2plotInterp[i]]
    # Plot response of likelihood neurons
    ap.plot_posterior_neuron(ax=ax, posteriorCtg=posteriorCtg,
                             ctgInd=ctgIndRep, ctgVal=ctgVal360,
                             trueVal=ctgVal360[ctg2plot[i]],
                             quantiles = quantiles,
                             meanOrMedian='median')
    ax.set_xlabel('Direction (deg)')
    # If it is first column, set y label
    ax.set_ylabel('Likelihood neuron response', fontsize=8)
    # Remove y ticks
    ax.set_yticks([])
    ax.set_xticks([0, 90, 180, 270, 360])
    # Put vertical lines at [-135, -45, 45, 135]
    ax.axvline(45, color='grey', alpha=0.5)
    ax.axvline(135, color='grey', alpha=0.5)
    ax.axvline(225, color='grey', alpha=0.5)
    ax.axvline(315, color='grey', alpha=0.5)
    # Set title
    ax.set_title(f'Selectivity: {ctgVal360[ctg2plot[i]]:.2f} deg', fontsize=10)
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}likelihood_neuron_dir_{ctgVal[ctg2plot[i]]:.2f}.png',
              bbox_inches='tight', pad_inches=0.01)
        plt.close()
    else:
        plt.show()



###############
# PLOT LIKELIHOOD NEURON TUNING CURVE FOR TRAIN STIMULI
###############

plotTypeDirName = f'{plotDirName}6_posteriors_train_stim/'
os.makedirs(plotTypeDirName, exist_ok=True)
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

# Make a category vector with literature angle convention
ctgVal360 = shift_angles(ctgVal)

# Compute posteriors
repeats = 5
posteriors = []
ctgIndRep = []
for r in range(repeats):
    print('Repeat: ', r)
    posteriors.append(ama.get_posteriors(s=s).detach())
    ctgIndRep.append(ctgInd)
posteriors = torch.cat(posteriors)
posteriors = posteriors.type(torch.float32)
ctgIndRep = torch.cat(ctgIndRep)

# Find the category indices of the stimuli in the interpolated values
ctgValInterp = ama.ctgVal
ctgIndInterp = find_interp_indices(ctgVal, ctgValInterp, ctgIndRep)

# Likelihood neuron selectivities to plot
#ctg2plot = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
ctg2plot = torch.arange(25)
ctg2plotInterp = (ctg2plot) * (interpPoints + 1)

quantiles = [0.40, 0.60]
# Plot the likelihood neurons
for i in range(len(ctg2plot)):
    fig, ax = plt.subplots(figsize=(2.2,1.5))
    # Class posteriors
    posteriorCtg = posteriors[:, ctg2plotInterp[i]]
    # Plot response of likelihood neurons
    ap.plot_posterior_neuron(ax=ax, posteriorCtg=posteriorCtg,
                             ctgInd=ctgIndRep, ctgVal=ctgVal360,
                             trueVal=ctgVal360[ctg2plot[i]],
                             quantiles = quantiles,
                             meanOrMedian='median')
    ax.set_xlabel('Direction (deg)')
    # If it is first column, set y label
    ax.set_ylabel('Likelihood neuron response', fontsize=8)
    # Remove y ticks
    ax.set_yticks([])
    ax.set_xticks([0, 90, 180, 270, 360])
    # Put vertical lines at [-135, -45, 45, 135]
    ax.axvline(45, color='grey', alpha=0.5)
    ax.axvline(135, color='grey', alpha=0.5)
    ax.axvline(225, color='grey', alpha=0.5)
    ax.axvline(315, color='grey', alpha=0.5)
    # Set title
    ax.set_title(f'Selectivity: {ctgVal360[ctg2plot[i]]:.2f} deg', fontsize=10)
    #posteriorStats = au.get_estimate_statistics(posteriorCtg, ctgIndRep,
    #                                           quantiles=quantiles)
    #multFactor = 1
    #ax.fill_between(ctgVal, posteriorStats['lowCI']*multFactor,
    #                 posteriorStats['highCI']*multFactor, color='black', alpha=0.1)
    #ax.plot(ctgVal, posteriorStats['estimateMean']*multFactor, color='black')
    ## If it is the first row, remove x ticks
    #ax.set_xlabel('Direction (deg)')
    ## If it is first column, set y label
    #ax.set_ylabel('Response')
    ## Remove y ticks
    #ax.set_yticks([])
    #ax.set_title(f'Selectivity: {ctgVal[ctg2plot[i]]:.2f} deg', fontsize=12)
    if savePlots:
        plt.savefig(fname=f'{plotTypeDirName}likelihood_neuron_dir_{ctgVal[ctg2plot[i]]:.2f}.png',
              bbox_inches='tight', pad_inches=0.01)
        plt.close()
    else:
        plt.show()








#
################
## PLOT ENERGY NEURON TUNING CURVES
################
#
#plotTypeDirName = f'{plotDirName}6bis_energy_literature_stim/'
#os.makedirs(plotTypeDirName, exist_ok=True)
#plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})
#
## Get responses
#repeats = 5
#responses = []
#ctgIndRep = []
#for i in range(repeats):
#    responses.append(ama.get_responses(s=sTst).detach())
#    ctgIndRep.append(ctgIndTst)
#responses = torch.cat(responses)
#ctgIndRep = torch.cat(ctgIndRep)
#
## Plot filter pairs as squared and added
#nFilters = 10
#
#respSq = responses**2
#for j in range(nFilters):
#    for i in range(j):
#        energyResp = respSq[:, i] + respSq[:, j]
#        energyStats = au.get_estimate_statistics(energyResp, ctgIndRep,
#                                                   quantiles=quantiles)
#        fig, ax = plt.subplots(figsize=(3.5,3.5))
#        multFactor = 1
#        ax.fill_between(ctgVal, energyStats['lowCI']*multFactor,
#                         energyStats['highCI']*multFactor, color='black', alpha=0.1)
#        ax.plot(ctgVal, energyStats['estimateMean']*multFactor, color='black')
#        # If it is the first row, remove x ticks
#        ax.set_xlabel('Direction (deg)')
#        # If it is first column, set y label
#        ax.set_ylabel('Response')
#        # Remove y ticks
#        ax.set_yticks([])
#        ax.set_title(f'Filter pair: {i}-{j}', fontsize=12)
#        plt.savefig(fname=f'{plotTypeDirName}1_sum_energy_neuron_filters_{i}-{j}.png',
#              bbox_inches='tight', pad_inches=0)
#        plt.close()
#
#
## Plot filter pairs as multiplied
#for j in range(nFilters):
#    for i in range(j):
#        energyResp = responses[:, i] * responses[:, j]
#        energyStats = au.get_estimate_statistics(energyResp, ctgIndRep,
#                                                   quantiles=quantiles)
#        fig, ax = plt.subplots(figsize=(3.5,3.5))
#        multFactor = 1
#        ax.fill_between(ctgVal, energyStats['lowCI']*multFactor,
#                         energyStats['highCI']*multFactor, color='black', alpha=0.1)
#        ax.plot(ctgVal, energyStats['estimateMean']*multFactor, color='black')
#        # If it is the first row, remove x ticks
#        ax.set_xlabel('Direction (deg)')
#        # If it is first column, set y label
#        ax.set_ylabel('Response')
#        # Remove y ticks
#        ax.set_yticks([])
#        ax.set_title(f'Filter pair: {i}-{j}', fontsize=12)
#        plt.savefig(fname=f'{plotTypeDirName}2_mult_energy_neuron_filters_{i}-{j}.png',
#              bbox_inches='tight', pad_inches=0)
#        plt.close()
#
#
