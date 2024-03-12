# <codecell>
##############
#### IMPORT PACKAGES
##############
import scipy.io as spio
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import ama_library.ama_class as cl
import ama_library.utilities as au
import ama_library.plotting as ap
from scipy.interpolate import CubicSpline
import einops
import time


def unpack_matlab_data(matlabData, ctgIndName='ctgInd', ctgValName='X'):
    """ Unpack the data from the matlab file into the appropriate
    format for the model.
    ----------------
    Arguments:
    ----------------
      - matlabData: Dictionary containing the data from the matlab file.
      - ctgIndName: Name of the field containing the category indices.
      - ctgValName: Name of the field containing the category values.
    ----------------
    Outputs:
    ----------------
      - s: Disparity stimuli. (nStim x nFilters)
      - ctgInd: Category index for each stimulus. (nStim x 1)
      - ctgVal: Values of the latent variable for each category. (nCtg x 1)
    """
    # Extract disparity stimuli
    if 's' in matlabData.keys():
      s = matlabData.get("s")
    else:
      s = matlabData.get("Iret")
    s = torch.from_numpy(s)
    s = s.transpose(0, 1)
    s = s.float()
    # Extract the vector indicating category of each stimulus row
    ctgInd = matlabData.get(ctgIndName)
    ctgInd = torch.tensor(ctgInd)
    ctgInd = ctgInd.flatten()
    ctgInd = ctgInd-1       # convert to python indexing (subtract 1)
    ctgInd = ctgInd.type(torch.LongTensor)  # convert to torch integer
    # Extract the values of the latent variable
    ctgVal = matlabData.get(ctgValName)
    ctgVal = torch.from_numpy(ctgVal)
    ctgVal = ctgVal.float()
#    ctgVal = ctgVal.flatten().float()
    return (s, ctgInd, ctgVal)


def contrast_stim(s, nChannels=1):
    """Take a batch of stimuli and convert to Weber contrast stimulus
    That is, subtracts the stimulus mean, and then divides by the mean.
    ----------------
    Arguments:
    ----------------
      - s: Stimuli batch. (nStim x nDimensions)
      - nChannels: Channels into which to separate the stimulus to make each
          channel into contrast individually.
    ----------------
    Outputs:
    ----------------
      - sContrast: Contrast stimulus. (nStim x nDimensions)
    """
    s_split = torch.chunk(s, nChannels, dim=1)
    s_contrast_split = []
    for s_part in s_split:
        sMean = torch.mean(s_part, axis=1)
        sContrast = torch.einsum('nd,n->nd', (s_part - sMean.unsqueeze(1)), 1/sMean)
        s_contrast_split.append(sContrast)
    sContrast = torch.cat(s_contrast_split, dim=1)
    return sContrast


def unvectorize_1D_binocular_video(inputVec, nFrames=15):
    """
    Take a 1D binocular video, in the shape of a vector, and
    reshape it into a 2D matrix, where each row is a time frame,
    each column is a time-changing pixel, and the left and right
    half of the matrix contain the left eye and right eye videos
    respectively.
    -----------------
    Arguments:
    -----------------
      - inputVec: Vector that contains a 1D binocular video. It
          can be  matrix, where each row is a 1D binocular video.
      - frames: Number of time frames in the video
    -----------------
    Outputs:
    -----------------
      - matVideo: 2D format of the 1D video, with rows as frames and
          columns as pixels. (nStim x nFrames x nPixels*2)
    """
    if inputVec.dim() == 1:
        inputVec = inputVec.unsqueeze(0)
    nVec = inputVec.shape[0]
    nPixels = round(inputVec.shape[1]/(nFrames*2))
    outputMat = torch.zeros(nVec, nFrames, nPixels*2)
    leftEye = inputVec[torch.arange(nVec), 0:(nPixels*nFrames)]
    rightEye = inputVec[torch.arange(nVec), (nPixels*nFrames):]
    leftEye = leftEye.reshape(nVec, nFrames, nPixels)
    rightEye = rightEye.reshape(nVec, nFrames, nPixels)
    outputMat = torch.cat((leftEye, rightEye), dim=2)
    return outputMat


def vectorize_2D_binocular_video(matVideo, nFrames=15):
    """
    Inverts the transformation of the unvectorize_1D_binocular_video function.
    Takes the 2D format of the binocular video and converts it back to its 1D form.
    -----------------
    Arguments:
    -----------------
      - matVideo: 2D format of the binocular video (nStim x nFrames x nPixels*2).
      - nFrames: Number of time frames in the video (default: 15).
    -----------------
    Outputs:
    -----------------
      - outputVec: 1D binocular video. It can also be a matrix, where each
          row is a 1D binocular video.
    """
    nStim = matVideo.shape[0]
    nPixels2 = matVideo.shape[2]
    nPixels = nPixels2 // 2  # nPixels for one eye
    # Split the left and right eyes
    leftEye = matVideo[:, :, :nPixels]
    rightEye = matVideo[:, :, nPixels:]
    # Reshape each eye tensor to 1D form
    leftEye = leftEye.reshape(nStim, nPixels * nFrames)
    rightEye = rightEye.reshape(nStim, nPixels * nFrames)
    # Concatenate the left and right eyes along the second dimension (columns)
    outputVec = torch.cat((leftEye, rightEye), dim=1)
    # If there's only one stimulus, we can squeeze to remove the first dimension
    if nStim == 1:
        outputVec = outputVec.squeeze(0)
    return outputVec


def subsample_categories_centered(nCtg, subsampleFactor):
    """
    Subsample the number of categories, while keeping the middle category.
    ----------------
    Arguments:
    ----------------
        - nCtg: Number of categories
        - subsampleFactor: Factor by which the categories will be subsampled
    ----------------
    Outputs:
    ----------------
        - subsampledInds: Vector containing the indices of the subsampled categories.
            These are equispaced with one another, and keep the middle category.
    """
    # Generate original vector
    allInds = np.arange(nCtg)
    # Ensure nCtg is odd
    assert len(allInds) % 2 == 1, "nCtg must be odd."
    # Find middle index
    midIdx = len(allInds) // 2
    # Calculate the start index for the left and right subsample
    start_left = midIdx % subsampleFactor
    start_right = midIdx + subsampleFactor
    # Subsample vector, maintaining the middle element
    subsampledInds = np.concatenate((allInds[start_left:midIdx:subsampleFactor], 
                                 allInds[midIdx:midIdx+1], 
                                 allInds[start_right::subsampleFactor]))
    return subsampledInds


def remove_categories(removeCtg, ctgVal, ctgInd, s):
    """ Remove the categories with indices in removeCtg,
    and reindex the category indices accordingly.
    ----------------
    Arguments:
    ----------------
      - removeCtg: Indices of the categories to remove.
      - ctgVal: Values of the latent variable for each category. (nCtg)
      - ctgInd: Category index for each stimulus. (nStim)
      - s: Disparity stimuli. (nStim x nFilters) 
    ----------------
    Outputs:
    ----------------
      - ctgValNew: Values of the latent variable for each category. (nCtgNew)
      - ctgIndNew: Category index for each stimulus. (nStim)
      - sNew: Disparity stimuli. (nStim x nFilters) 
    """
    # Remove categories and update ctgVal
    ctgValNew = np.delete(ctgVal, removeCtg, axis=0)
    # Find elements in ctgInd that are in removeCtg
    keepInds = np.where(~np.isin(ctgInd, removeCtg))[0]
    # Remove corresponding elements from ctgInd
    ctgIndNew = ctgInd[keepInds]
    # Remove corresponding rows from s
    sNew = s[keepInds, :]
    # Remap old indices to new categories
    ctgIndNew = reindex_categories(ctgIndNew)
    return ctgValNew, ctgIndNew, sNew


def reindex_categories(ctgIndNew):
    """ Reindex the category indices so that they are consecutive."""
    unique_categories = np.unique(ctgIndNew)
    category_mapping = {old_index: new_index for new_index,
                        old_index in enumerate(unique_categories)}
    for old_index, new_index in category_mapping.items():
        ctgIndNew[ctgIndNew == old_index] = new_index
    return ctgIndNew


def subsample_cov_inds(covariance, keepInds):
    """ Subsample the covariance matrix, keeping only the indices in keepInds.
    ----------------
    Arguments:
    ----------------
      - covariance: Covariance matrix to subsample. (nCtg x nCtg)
      - keepInds: Indices of the categories to keep.
    ----------------
    Outputs:
    ----------------
      - covarianceNew: Subsampled covariance matrix. (nCtgNew x nCtgNew)
    """
    covarianceNew = covariance[:,keepInds,:]
    covarianceNew = covarianceNew[:,:,keepInds]
    return covarianceNew


def linear_interpolation(y, x=None, nPoints=1):
      """ Interpolate between a set of vectors using linear interpolation.
      -----------------
      Arguments:
      -----------------
      - y: Array containing the values to interpolate
      - x: Array containing the x values of the y's
      - nPoints: Number of points to interpolate between each pair of
        y values.
      -----------------
      Outputs:
      -----------------
      - yInterp: Array containing the interpolated values.
      """
      n = y.shape[0]
      if x is None:
          x = np.linspace(start=0, stop=nPoints-1, num=n)
      yInterp = np.zeros((n-1)*(nPoints+1)+1)
      for i in range(n-1):
          xInterp = np.linspace(start=x[i], stop=x[i+1], num=nPoints+2)
          yInterp[i*(nPoints+1):((i+1)*(nPoints+1)+1)] = np.interp(xInterp, x[i:(i+2)],
                                                                   y[i:(i+2)])
      return torch.tensor(yInterp, dtype=torch.float32)


def spline_interpolation(y, x=None, nPoints=1):
      """ Interpolate between a set of vectors using cubic splines.
      -----------------
      Arguments:
      -----------------
      - y: Array containing the values to interpolate
      - x: Array containing the x values of the y's
      - nPoints: Number of points to interpolate between each pair of
        y values.
      -----------------
      Outputs:
      -----------------
      - yInterp: Array containing the interpolated values.
      """
      n = y.shape[0]
      if x is None:
          x = np.linspace(start=0, stop=nPoints-1, num=n)
      splineInterp = CubicSpline(x=x, y=y, axis=0)
      yInterp = np.zeros((n-1)*(nPoints+1)+1)
      for i in range(n-1):
          xInterp = np.linspace(start=x[i], stop=x[i+1], num=nPoints+2)
          yInterp[i*(nPoints+1):((i+1)*(nPoints+1)+1)] = splineInterp(xInterp)
      return yInterp


# test spline interpolation
#x = np.linspace(start=-1, stop=1, num=10)
#y = x**2
#yInterp = spline_interpolation(y, x, nPoints=5)
#xInterp = spline_interpolation(x, x, nPoints=5)
#plt.plot(yInterp, 'o')


def covariance_interpolation(covariance, nPoints=1):
    """ Interpolate between a set of covariance matrices using cubic splines.
    -----------------
    Arguments:
    -----------------
    - covariance: Array or tensor containing the covariances, of shape
      (nCtg, nDim, nDim).
    - nPoints: Number of points to interpolate between each pair of
      covariances.
    -----------------
    Outputs:
    -----------------
    - covarianceInterp: Array containing the interpolated covariances, of shape
        (nCtg * (nPoints + 1), nDim, nDim).
    """
    nCovs, nDim, _ = covariance.shape
    covarianceInterp = np.zeros(((nCovs - 1) * (nPoints) + nCovs, nDim, nDim))
    for i in range(nDim):
        for j in range(nDim):
            covarianceInterp[:, i, j] = spline_interpolation(covariance[:, i, j],
                                                              nPoints=nPoints)
            covarianceInterp[:, j, i] = covarianceInterp[:, i, j]
    return torch.tensor(covarianceInterp, dtype=torch.float32)


def mean_interpolation(mean, nPoints=1):
    """ Interpolate between a set of mean vectors using cubic splines.
    -----------------
    Arguments:
    -----------------
    - mean: Array or tensor containing the means, of shape
      (nCtg, nDim).
    - nPoints: Number of points to interpolate between each pair of
      means.
    -----------------
    Outputs:
    -----------------
    - meanInterp: Array containing the interpolated means, of shape
        (nCtg * (nPoints + 1), nDim).
    """
    nCovs, nDim = mean.shape
    meanInterp = np.zeros(((nCovs - 1) * (nPoints) + nCovs, nDim))
    for i in range(nDim):
        meanInterp[:, i] = spline_interpolation(mean[:, i], nPoints=nPoints)
    return torch.tensor(meanInterp, dtype=torch.float32)


def interpolate_circular_ama(ama, interpPoints=1):
    """ Interpolate the statistics of a circular variable AMA model.
    It prepends and appends the statistics of the model to itself, and then
    interpolates. Thus, the first and last categories do not have a border
    effect, but are adjacent to repeats of each other.
    -----------------
    Arguments:
    -----------------
      - ama: AMA model object
      - interpPoints: Number of points to interpolate between each pair of
        categories.
    -----------------
    Outputs:
    -----------------
      - ama: AMA model object with interpolated statistics.
    """
    # Interpolate class statistics
    # First, triplicate the statistics along first (ctg) dimension, so splines
    # work with the circular variable. Use einops
    nCtg = ama.ctgVal.shape[0]
    respCovDup = einops.repeat(ama.respCov.detach(), 'c x y -> (c1 c) x y', c1=3)
    respMeanDup = einops.repeat(ama.respMean.detach(), 'c x -> (c1 c) x', c1=3)
    ctgVal = ama.ctgVal.detach()
    ctgValDup = torch.cat((ctgVal-360, ctgVal, ctgVal+360))
    # Interpolate
    respCovDup = covariance_interpolation(covariance=respCovDup,
                                          nPoints=interpPoints)
    respMeanDup = mean_interpolation(mean=respMeanDup, nPoints=interpPoints)
    ctgValDup = torch.tensor(linear_interpolation(y=ctgValDup, nPoints=interpPoints),
                             dtype=torch.float32)
    p1 = int(nCtg*(interpPoints+1))
    p2 = int((respCovDup.shape[0] - p1 + interpPoints + 1)/2)
    respCovInt = respCovDup[p2:(p1+p2),:,:]
    respMeanInt = respMeanDup[p2:(p1+p2),:]
    ctgValInt = ctgValDup[p2:(p1+p2)]
    # Assign to model
    ama.respCov = respCovInt
    ama.respMean = respMeanInt
    ama.ctgVal = ctgValInt
    return ama


def plot_estimate_statistics(ax, estMeans, errorInterval, ctgVal=None, color='b'):
    """ Plot the estimated mean and confidence interval of the
    model estimates at each value of the latent variable.
    -----------------
    Arguments:
    -----------------
      - estMeans: Tensor containing the mean (or median) value
          of the model estimates for each x
      - errorInterval: Tensor containing the standard deviation of the
          model estimates for each x
      - ctgVal: Tensor containing the value of the latent variable
          for each x. If None, then it is assumed that the latent
          variable is a linearly spaced vector between -1 and 1.
    """
    if ctgVal is None:
        ctgVal = torch.linspace(-1, 1, len(estMeans))
    if not torch.is_tensor(ctgVal):
        ctgVal = torch.tensor(ctgVal)
    if errorInterval.dim() == 1:
        errorInterval = sd_to_ci(means=estMeans, sd=errorInterval,
                                 multiplier=1)
    # convert to numpy for matplotlib compatibility
    estMeans = estMeans.detach().numpy()
    ctgVal = ctgVal.detach().numpy()
    plt.plot(ctgVal, estMeans, color=color, linewidth=4)
    plt.fill_between(ctgVal, errorInterval[0,:], errorInterval[1,:],
                     color=color, alpha=0.2)
    plt.axline((0,0), slope=1, color='black', linestyle='--', linewidth=2)


def find_interp_indices(ctgVal, ctgValInterp, ctgInd):
    ctgIndNew = torch.zeros(ctgInd.shape).type(torch.long)
    for indOld in range(len(ctgVal)):
        indNew = torch.where(ctgValInterp==ctgVal[indOld])[0]
        ctgIndNew[ctgInd==indOld] = int(indNew)
    return ctgIndNew


def remove_categories_stats(statsDict, inds2keep):
    """ Remove the categories with indices in removeCtg,
    and reindex the category indices accordingly.
    ----------------
    Arguments:
    ----------------
      - statsDict: Dict of statistics to remove categories from.
      - inds2keep: Indices of the categories to keep.
    ----------------
    Outputs:
    ----------------
      - statsDictNew: List of statistics with the categories removed.
    """
    dictKeys = list(statsDict.keys())
    statsDictNew = {}
    for i in range(len(dictKeys)):
        statsDictNew[dictKeys[i]] = statsDict[dictKeys[i]][inds2keep]
    return statsDictNew


def match_circular_stats(ctgVal, estimateStats):
    """ For estimates of an angle, make the numbers coherent by choosing the
    angle that is closest to the corresponding ctgVal. For example, if the
    mean estimate for ctgVal=355 is 10, then this function will change it to
    370, since this is closer, linearly, to 355. The confidence intervals
    are also adjusted accordingly.
    This function works for the output of AMA circular variable estimate
    statistics, it's not general purpose.
    ----------------
    Arguments:
    ----------------
      - ctgVal: Values of the latent variable for each category. (nCtg x 1)
      - estimateStats: Dictionary containing the statistics of the estimates
          for each category. Has fields estimateMean, estimateMedian, lowCI,
          highCI. Each field has a vector of length nCtg.
    ----------------
    Outputs:
    ----------------
      - estimateStats: Dictionary containing the statistics of the estimates
          for each category. (nCtg x 1)
    """
    statType = ['Median', 'Mean']
    for k in range(len(ctgVal)):
        for stat in statType:
            # Check if mean is closer to ctgVal[k] or ctgVal[k]-360
            dist1 = torch.abs(estimateStats[f'estimate{stat}'][k] - ctgVal[k])
            dist2 = torch.abs(estimateStats[f'estimate{stat}'][k] - ctgVal[k] - 360)
            if dist1 > dist2:
                estimateStats[f'estimate{stat}'][k] = estimateStats[f'estimate{stat}'][k] - 360
                estimateStats[f'lowCI{stat}'][k] = estimateStats[f'lowCI{stat}'][k] - 360
                estimateStats[f'highCI{stat}'][k] = estimateStats[f'highCI{stat}'][k] - 360
    # Fix some edge cases
    if ctgVal[k]>300 and estimateStats[f'estimate{stat}'][k] < 100:
        estimateStats[f'estimate{stat}'][k] = estimateStats[f'estimate{stat}'][k] + 360
        estimateStats[f'lowCI{stat}'][k] = estimateStats[f'lowCI{stat}'][k] + 360
        estimateStats[f'highCI{stat}'][k] = estimateStats[f'highCI{stat}'][k] + 360
    return estimateStats


def shift_angles(x):
    # Turn angles from -180/180 to 0/360
    return torch.remainder(x + 270, 360)


def polar_2_Z(ctgVal2D):
    # Convert the 2D latent variable of speed and angle into
    # an only speed latent variable
    dirCos = torch.cos(torch.deg2rad(ctgVal2D[1,:]))
    dirCos[dirCos.isnan()] = 1
    ctgVal = ctgVal2D[0,:] * dirCos
    return ctgVal


