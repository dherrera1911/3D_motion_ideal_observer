function [LccdMV, RccdMV] = Iccd2BVS3D(LccdBffr, RccdBffr, PszXY, ...
  spdDegPerSecL, spdDegPerSecR, smpPerDeg, smpPerSec, durationMs, ...
  zeroDspTime, bPLOT)

% function [Lret,Rret,Lccd,Rccd,LretDC,RretDC,LccdDC,RccdDC] = Iccd2IretS3D(LccdBffr, RccdBffr, IPDm, PszXY, spdDegPerSecL, ...
% spdDegPerSecR, smpPerDeg, smpPerSec, durationMs, lensInfo, PSF, TIR, wndwInfo, W, bPreWndw, slntDeg, bPLOT)
%
%   example call: % LOAD LUMINANCE IMAGE
%                 [Lpht,Rpht,Lrng,Rrng,Lxyz,Rxyz,LppXm,LppYm,RppXm,RppYm]=loadLRSIimage(62,1,0,'PHT','img'); LphtBffr = cropImageCtr(Lpht,[],[600 120]); RphtBffr = cropImageCtr(Lpht,[],[600 120]);
%                 PszXY = [60 60]; W = cosWindowXYT([PszXY 15]); lensInfo = lensInfoStruct({'NVR' 'NVR'},4,'IDP','FLT',60,[PszXY],1);
%                 Iccd2IretS3D(LphtBffr,RphtBffr,0.065,[60 60],-1,1,60,60,250,[],1,[],[],W,0,0,1);
%
%                 IctrRC = [1000 500]; PszXYbffr = [360 120]; PszXY = [60 60];
%                 [~,~,~,~,LphtBffr,~,~,RphtBffr]=LRSIsamplePatch(8,'L',IctrRC,[PszXYbffr],-5,'PHT','linear',1,1);
%                 Iccd2IretS3D(LphtBffr,RphtBffr,0.065,[PszXY],-1,1,60,60,250,lensInfo,[],1,[],W,1,0,1);
%
% LccdBffr:      left  image with spatial buffer
% RccdBffr:      right image with spatial buffer
% IPDm:          interpupillary distance
% PszXY:         size of patch in pixels                            [1 x 2]
% spdDegPerSecL:  L eye speed of motion
% spdDegPerSecR:  R eye speed of motion
% smpPerDeg:     spatial sampling rate  of output movie (smp/deg)
% smpPerSec:     temporal sapmling rate of output movie (smp/sec = htz)
% durationMs:    duration in milliseconds
% PSF:           optical transfer function for left image
% TIR:           temporal impulse response function
% W:             space-time window (dimensions must match [ PszXY durationMs*smpPerSec/1000 ]
% bPreWndw:      boolean to indicate whether to pre-window the stimulus
%                1 -> bakes window into stimulus (good for psychophysics)
%                0 -> does not                   (good for ...)
% slntDeg:       slant about the x-axis (tilt = 90deg)
% bPLOT:         1 -> plot movie
%                0 -> don't
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lret:         retinal        image in column vector form
% Rret:         retinal        image in column vector form
% Lccd:         onscreen movie image in column vector form
% Rccd:         onscreen movie image in column vector form
% LccdDC:          mean luminance of left  eye movie
% RccdDC:          mean luminance of right eye movie

% INPUT HANDLING
if ~exist('RccdBffr','var') || isempty(RccdBffr)
    LccdBffr = RccdBffr;
end
if ~exist('smpPerSec','var') || isempty(smpPerSec)
    smpPerSec = 480;
end
if size(LccdBffr,1) < PszXY(2) || size(LccdBffr,2) < PszXY(1)
    error(['Iccd2IretS3D: WARNING! LccdBffr smaller than PszXY']);
end

% INPUT HANDLING: CHECK BUFFER SIZE
PszRCbffr = size(LccdBffr,1)*[1 1];
PszRC = fliplr(PszXY);
PszRCdff = PszRCbffr-PszRC;
if mod(PszRCdff(1),2) ~= 0 || mod(PszRCdff(2),2) ~= 0
    error(['Iccd2IretS3D: WARNING! PszXYbffr differs from PszXY by odd number of pixels! Fix it!']);
end

%%%%%%%%%%%%%%%%%%%%%
% TEMPORAL SAMPLING %
%%%%%%%%%%%%%%%%%%%%%

% GET THE NUMBER OF TEMPORAL SAMPLES
numSmp = durationMs*smpPerSec/1000;
% GET THE SPATIAL TRANSLATION PER SAMPLE IN PIXELS
msPerSmp       = 1000/smpPerSec;  % How many ms are in a sample
% Speed in pixels per Ms
pixDffPerMsL       = (spdDegPerSecL.*smpPerDeg)./1000;
pixDffPerMsR       = (spdDegPerSecR.*smpPerDeg)./1000;
% Speed in pixels per sample
pixDffPerSmpL = pixDffPerMsL.*msPerSmp;
pixDffPerSmpR = pixDffPerMsR.*msPerSmp;

%%%%%%%%%%%%%
% CCD MOVIE %
%%%%%%%%%%%%%
PszXYbffrSq = size(LccdBffr,1)*[1 1];

[LccdMV, LmovieCoords] = motionImagesCrop(LccdBffr, PszXY, pixDffPerSmpL, ...
  numSmp, [], 0, 'zeroDspTime', zeroDspTime);
[RccdMV, RmovieCoords] = motionImagesCrop(RccdBffr, PszXY, pixDffPerSmpR, ...
  numSmp, [], 0, 'zeroDspTime', zeroDspTime);

%%%%%%%%%%%%%%%%
% PLOT RESULTS %
%%%%%%%%%%%%%%%%
if ~exist('bPLOT','var') || isempty(bPLOT)
    bPLOT = 0;
end
if bPLOT
    %%
    figure(12121);
    set(gcf,'position',[600 150 500 600]);
    for i = 1:size(LccdMV,length(size(LccdMV)))
        imagesc([LccdMV(:,:,i) RccdMV(:,:,i); LccdMV(:,:,i) RccdMV(:,:,i)]);
        axis image; axis xy;
        colormap gray;
        caxis(minmax([LccdMV RccdMV]));
        formatFigure('RET',[],'CCD')
        set(gca,'xtick',[]); set(gca,'ytick',[]);
        % pause(5.*msPerSmp./1000);
        pause(.08);
    end
end
