##########################################
# This script trains an AMA model on the task of 3D direction estimation.
# The script takes as inputs a .mat file containing the vertically
# averaged 3D motion stimuli and the corresponding speed labels.
# The characteristics of the stimuli are specified in its file name.
# The parameters for model training are specified in the script
# by the user.
#
# Code author: Daniel Herrera-Esposito, dherrera1911 at gmail dot com
##########################################


##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
import sys
import os
sys.path.append('../../code/')
from auxiliary_functions import *
import time

start = time.time()

##############
#### LOAD DATA
##############

dataDir = '../../data/'

# SPECIFY WHAT DATASET TO LOAD
downSample = 2  # Factor by which pixels are downsampled in input stimuli
spd = '0.15'  # Max speed in input stimuli
degStep = '7.5'   # Step in deg between classes
looming = '1'  # Whether looming is included in stimuli
dspStd = '00'

# LOAD DATA
# Training
data = spio.loadmat(dataDir + 'ama_inputs/'
  f'D3D-nStim_0500-spd_{spd}-degStep_{degStep}-dspStd_{dspStd}-'
  f'dnK_{downSample}-loom_{looming}-TRN.mat')
s, ctgInd, ctgVal = unpack_matlab_data(
    matlabData=data, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Testing
dataTst = spio.loadmat(dataDir + 'ama_inputs/'
  f'D3D-nStim_0300-spd_{spd}-degStep_{degStep}-dspStd_{dspStd}-'
  f'dnK_{downSample}-loom_{looming}-TST.mat')
sTst, ctgIndTst, ctgValTst = unpack_matlab_data(
    matlabData=dataTst, ctgIndName='ctgIndMotion', ctgValName='Xmotion')
# Extract some properties of the dataset
nStim = s.shape[0]
df = s.shape[1]

##############
#### SET TRAINING PARAMETERS FOR THE MODEL
##############

# NOISE PARAMETERS
respNoiseVec = torch.tensor([0.01])
pixelSigmaEqv = 3.75*10**-4
pixelNoiseSigma = au.noise_total_2_noise_pix(sigmaEqv=pixelSigmaEqv,
                                             numPix=df/2)
pixelNoiseVar = pixelNoiseSigma**2
# NUMBER OF FILTERS AND SEEDS
nPairs = 5   # Number of filters to use
seedsByPair = 7
# TRAINING EPOCHS AND BATCHES
nEpochs = 100
batchSize = 2048
# LOSS
#lossFun = au.cross_entropy_loss
lossFun = au.kl_loss
# LEARNING RATE
regime = 'step' # 'one-cycle' or 'step'
initLR = 0.02  # Initial learning rate (both regimes)
lrGamma = 0.8   # Multiplication factor for lr decay (step regime)
lrStepSize = 10  # Number of epochs between lr decay (step regime)

##############
#### PREPROCESS DATA
##############

# Convert indices and categories to Z-motion speeds
ctgVal = ctgVal[1,:]
ctgValTst = ctgValTst[1,:]
# Convert intensity stimuli to contrast stimuli
s = contrast_stim(s=s, nChannels=2)
sTst = contrast_stim(s=sTst, nChannels=2)

# Move data to available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
s = s.to(device)
ctgInd = ctgInd.to(device)
ctgVal = ctgVal.to(device)
sTst = sTst.to(device)
ctgIndTst = ctgIndTst.to(device)
ctgValTst = ctgValTst.to(device)

# Put data into Torch data loader tools
trainDataset = TensorDataset(s, ctgInd)
# Batch loading and other utilities 
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

# Function that returns an optimizer
def opt_fun(model):
    return torch.optim.Adam(model.parameters(), lr=initLR)
# Function that returns a scheduler
def scheduler_fun(opt):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=lrStepSize,
                                           gamma=lrGamma)

##############
#### TRAIN MODELS
##############

for rn in range(len(respNoiseVec)):
    respNoiseVar = respNoiseVec[rn]
    ##############
    #### INITIALIZE AND TRAIN MODEL
    ##############

    # Initialize desired AMA variant
    samplesPerStim = 10
    ama = cl.AMA_emp(sAll=s, ctgInd=ctgInd, nFilt=2, respNoiseVar=respNoiseVar,
            pixelCov=pixelNoiseVar, ctgVal=ctgVal,
            samplesPerStim=samplesPerStim, nChannels=2, device=device)

    # Train
    trainLoss, testLoss, elapsedTime, filterList = au.fit_by_pairs(
        nEpochs=nEpochs, model=ama, trainDataLoader=trainDataLoader,
        lossFun=lossFun, opt_fun=opt_fun, nPairs=nPairs,
        scheduler_fun=scheduler_fun, seedsByPair=seedsByPair,
        sTst=sTst, ctgIndTst=ctgIndTst, printProg=True)

    trainingDict = {
        'nEpochs': nEpochs, 'batchSize': batchSize,
        'nPairs': nPairs, 'seedsByPair': seedsByPair,
        'respNoiseVar': respNoiseVar, 'pixelNoiseVar': pixelNoiseVar,
        'lrGamma': lrGamma, 'lrInitial': initLR,
        'lrStepSize': lrStepSize, 'lossFun': 'KL',
        'trainLoss': trainLoss, 'testLoss': testLoss,
        'time': elapsedTime, 'filterList': filterList,
        'filters': ama.f.detach().clone(),
        'respCov': ama.respCov.detach().clone(),
        'respCovNoiseless:': ama.respCovNoiseless.detach().clone()}

    import torch.nn.utils.parametrize as parametrize
    ama.to('cpu')
    parametrize.remove_parametrizations(ama, "f", leave_parametrized=True)
    dirName = dataDir + 'trained_models/'
    # If directory does not exist, create it
    if not os.path.exists(dirName):
        os.makedirs(dirName, exist_ok=True)
    fileName = f'ama_3D_dir_empirical_dnK_{downSample}_spd_{spd}_' + \
      f'degStep_{degStep}_noise_{respNoiseVar:.4f}_loom_{looming}_dspStd_{dspStd}.pt'
    torch.save(ama, dirName + fileName)
    fileName = f'ama_3D_dir_empirical_dnK_{downSample}_spd_{spd}_' + \
      f'degStep_{degStep}_noise_{respNoiseVar:.4f}_loom_{looming}_dspStd_{dspStd}_dict.pt'
    torch.save(trainingDict, dirName + fileName)

end = time.time()

elapsedTime = end - start
minutes, seconds = divmod(int(elapsedTime), 60)
print(f'Full process took {minutes:02d}:{seconds:02d}')

