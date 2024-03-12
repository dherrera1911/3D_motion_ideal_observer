##########################################
# This script plots some training parameters for the 3D
# speed estimation model. For example, it plots the loss
# as a function of the number of epochs, and it shows the
# agreement between different seeds of the training.
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
import copy
import os

savePlots = True
spdStep = '0.100'
maxSpd = '2.50'
loom = '0'
dspStd = '00'
for dnK in [2]:
    for noise in ['0.0100']:
        # Check if a file with this characteristics exists
        modelFile = f'./data/trained_models/' \
            f'ama_3D_speed_empirical_dnK_{dnK}_maxSpd_{maxSpd}_spdStep_{spdStep}_noise_{noise}' \
            f'_loom_{loom}_dspStd_{dspStd}_dict.pt'
        if os.path.isfile(modelFile):
            plotDirName = f'./plots/3D_speed_training/dnK{dnK}_spd{maxSpd}_' \
              f'noise{noise}_spdStep{spdStep}/'
            os.makedirs(plotDirName, exist_ok=True)
            # Load the model
            trainingDict = torch.load(modelFile)

            # Extract relevant variables
            nPairs = trainingDict['nPairs']
            nEpochs = trainingDict['nEpochs']
            trainingTimes = trainingDict['time']
            seedsByPair = trainingDict['seedsByPair']
            trainLoss = trainingDict['trainLoss']
            testLoss = trainingDict['testLoss']

            ###############
            # 1) Plot and export the filters from each seed
            ###############

            filterList = trainingDict['filterList']

            plt.figure(figsize=(18, 7))
            for n in range(nPairs):
                for s in range(seedsByPair):
                    plt.subplot(nPairs, seedsByPair, n*seedsByPair + s + 1)
                    filterIm = unvectorize_1D_binocular_video(filterList[n][s], nFrames=15)
                    filterCat = torch.cat((filterIm[0], filterIm[1]), dim=1)
                    plt.imshow(filterCat, cmap='gray')
                    # If first row, add title
                    if n == 0:
                        plt.title(f'Seed {s+1}')
                    # If first column, add ylabel 
                    if s == 0:
                        plt.ylabel(f'Pair {n+1}')
                    # Remove ticks
                    plt.xticks([])
                    plt.yticks([])

            if savePlots:
                fileName = f'{plotDirName}1_filters_across_seeds.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()
            else: 
                plt.show()


            ###############
            # 2) Plot the train and test curve loss of each seed
            ###############

            plt.figure(figsize=(10, 13))
            for n in range(nPairs):
                plt.subplot(nPairs, 1, n+1)
                lossTrans = np.transpose(trainLoss[n])
                for s in range(seedsByPair):
                  plt.plot(trainLoss[n][s,:], label=f'{s+1}')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

            if savePlots:
                fileName = f'{plotDirName}2_training_loss.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()
            else: 
                plt.show()


            plt.figure(figsize=(10, 10))
            for n in range(nPairs):
                plt.subplot(nPairs, 1, n+1)
                lossTrans = np.transpose(testLoss[n])
                for s in range(seedsByPair):
                  plt.plot(testLoss[n][s,:], label=f'{s+1}')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

            if savePlots:
                fileName = f'{plotDirName}3_testing_loss.png'
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0)
                plt.close()
            else: 
                plt.show()


