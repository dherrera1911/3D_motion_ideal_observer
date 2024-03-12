function fname = buildFilenameLRSIpatchBV_XZ(natORflt, stmPerLvlDTB, PszXY, ...
  tgtSpdMeter, tgtDirDeg, tgtZPosMeter, dspStdArcMin)

% function fname = buildFilenameLRSIpatchBV(natORflt,numImg,stmPerLvlDTB,PszXY,spdDegPerSecL,spdDegPerSecR,projInfo,lensInfo,sensInfo,wndwInfo,bPreWndw,rndSdInfo)
%
%   example call: fname = buildFilenameLRSIpatchBV('FLT',95,2000,[60 60],2,-2,projInfo,lensInfo,sensInfo,wndwInfo,bPreWndw,rndSdInfo)
%                
% builds filename for saving and loading monocular luminance videos
% 
% natORflt:         string specifying if patches are planar or have depth structure
%                   'NAT'
%                   'FLT'
% numImg:           number of images
% stmPerLvlDTB:     number of stimuli in the dataset
% PszXY:            patch size
% spdDegPerSecL:    LE speed for this batch of binocular videos
% spdDegPerSecR:    RE speed for this batch of binocular videos
% projInfo:         project info struct with fields
% .spdDegPerSecLAll: speed in degrees per second... see projInfoStruct.m
% .smpPerDeg:       sampling rate in deg
% .smpPerSec:       sampling rate in sec (i.e. htz)
% .durationMs:      duration in milliseconds
% .slntPriorType:   type of slant prior
%                   'FSP' -> flat   slant prior   (i.e. all surface frontoparllel)
%                   'CSP' -> cosine slant prior   (i.e. truncated cosine disttribution of slants)
%                   'USP' -> unifornm slant prior (i.e. all slants equally likely) 
% lensInfo:         lensInfo  structure... see lensInfoStruct.m 
% sensInfo:         sensInfo  structure... see sensInfoStruct.m 
% wndwInfo:         wndInfo   strucutre... see wndwInfoStruct.m 
% bPreWndw:         boolean to indicate whether to pre-window the stimulus
%                   1 -> bakes window into stimulus (good for psychophysics)
%                   0 -> does not                   (good for ...)
% rndSdInfo:        rndSdInfo structure... see rndSdInfo.m
%%%%%%%%%%%%%%%%%%%%%%%
% fname:         filename with which to save or load images

fname = ['LRSI_MotionXZ_' natORflt '-nStim_' num2str(stmPerLvlDTB,'%04d')  ...
         '-imSize_' num2str(PszXY(1),'%03d') 'x' num2str(PszXY(2),'%03d')  ... 
         '-speed_' num2str(tgtSpdMeter,'%.3f') '-dir_' num2str(tgtDirDeg) ...
         '-zPos_' num2str(tgtZPosMeter,'%.2f') ...
         '-dspStd_', num2str(dspStdArcMin, '%02d'), '.mat'];

