function fname = buildFilenameS3D_Z(stmPerLvl, spdStep, maxSpd, dspStdArcMin, dnK, trnORtst)

% function fname = buildFilenameLRSIpatchS3D(natORflt,numImg,stmPerLvlDTB,PszXY,projInfo,lensInfo,sensInfo,wndwInfo,bPreWndw,imgDim,dnK,spdDegPerSecAll,stmPerLvl,rndSdInfo,trnORtst)
%                                            
%   example call: fname = buildFilenameLRSIpatchS3D('NAT',95,5000,[52 52],projInfo,lensInfo,sensInfo,wndwInfo,1,'2D',4,linspace(-8,8,21),1000,'TRN')
%
% builds filename for saving/loading ground-truth disparity-annotated patch sets
% 
% natORflt:         string indicating whether binocular patches will be sampled with or without depth structure
%                   'NAT' -> patches with natural depth structure
%                   'FLT' -> planar patches without natural depth structure
% numImg:           number of LRSI images from which the corresponding-points database is obtained
% stmPerLvlDTB:     number of patches per disparity in the saved BI structure
% PszXY:            patch-size in pixels  
% imgDim:           image dimensionality
%                   '2D' -> two dimensional (XY)
%                   '1D' -> one dimensional (X) vertical average
%                   'RD' -> one dimensional (X) radial   average
% dnK:              downsampling factor
% projInfo:
% lensInfo:
% sensInfo:
% wndwInfo:
% bPreWndw:         bakes window into stimulis
% spdDegPerSecAll:  vector of disparities for which patches will be added to the output structure
% stmPerLvl:        number of binocular patches per disparity in the saved structure
% rndSdInfo:   
% trnORtst:         string indicating whether the annotated image set saved will be used as a training or a test set
%                   'TRN' : Training set
%                   'TST' : Test set
% %%%%%%%%%%%%%%%%%%%%%%
% fname:            filename with which to save or load images

fname = ['S3D', '-nStim_', num2str(stmPerLvl,'%04d'), ...
         '-spdStep_', num2str(spdStep, '%.3f\n'), ...
         '-maxSpd_', num2str(maxSpd, '%.2f\n'), ...
         '-dspStd_', num2str(dspStdArcMin, '%02d'), ...
         '-dnK_', num2str(dnK), '-', trnORtst '.mat'];

