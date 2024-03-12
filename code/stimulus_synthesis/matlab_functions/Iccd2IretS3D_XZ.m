function [Lret, Rret, Lccd, Rccd, LretDC, RretDC, LccdDC, RccdDC] = Iccd2IretS3D_XZ(LccdBffr, ...
  RccdBffr, IPDm, PszXY, spdDegPerSecL, spdDegPerSecR, smpPerDeg, smpPerSec, durationMs, ...
  zeroDspTime, lensInfo, PSF, TIR, wndwInfo, W, bPreWndw, slntDeg, bPLOT)

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
if mod(smpPerSec.*durationMs/1000,1) ~= 0
    error(['Iccd2IretS3D: WARNING! durationMs must be whole number of samples in time. Currently, smpPerSec*durationMs/1000= ' num2str(smpPerSec*durationMs/1000)]);
end
if ~exist('wndwInfo','var') || isempty(wndwInfo)
    wndwInfo = [];
end
if ~exist('W') || isempty(W)
    W = ones([fliplr(PszXY) durationMs.*smpPerSec./1000] );
end
if size(LccdBffr,1) < PszXY(2) || size(LccdBffr,2) < PszXY(1)
    error(['Iccd2IretS3D: WARNING! LccdBffr smaller than PszXY']);
end
% CHECK WINDOW SIZE
if size(W,1) ~= PszXY(2) ||size(W,2) ~= PszXY(1)
    error(['Iccd2IretS3D: WARNING! invalid window size [' num2str(size(W,2)) 'x' num2str(size(W,1)) ']. PszXY [' num2str(PszXY(1)) 'x' num2str(PszXY(2)) ']']);
end

% INPUT HANDLING: CHECK BUFFER SIZE
PszRCbffr = size(LccdBffr,1)*[1 1];
PszRC = fliplr(PszXY);
PszRCdff = PszRCbffr-PszRC;
if mod(PszRCdff(1),2) ~= 0 || mod(PszRCdff(2),2) ~= 0
    error(['Iccd2IretS3D: WARNING! PszXYbffr differs from PszXY by odd number of pixels! Fix it!']);
end

%%%%%%%%%%%%%%%%%%%%
% SPATIAL SAMPLING %
%%%%%%%%%%%%%%%%%%%%
[Xdeg, Ydeg] = meshgrid( smpPos( smpPerDeg, PszXY(1)), smpPos(smpPerDeg, PszXY(2)));

%%%%%%%%%%%%%%%%%%%%%
% TEMPORAL SAMPLING %
%%%%%%%%%%%%%%%%%%%%%
% MINIMIUM HTZ FOR HI-RES SAMPLING
smpPerSecHiRes = 480;
% TEMPORAL SAMPLING RATE     OF OUTPUT MOVIE
smpPerMs       = smpPerSec./1000;
% MILLISECONDS PER SAMPLES   OF OUTPUT MOVIE
msPerSmp       = 1./smpPerMs;
% NUMBER OF TEMPORAL SAMPLES IN OUTPUT MOVIE
numSmp         = durationMs.*smpPerMs;
% UP-SAMPLE FACTOR TO HIGH RESOLUTION
upK            = ceil(smpPerSecHiRes./smpPerSec);
numSmpHiRes    = numSmp.*upK;
smpPerMsHiRes  = smpPerSecHiRes./1000;
msPerSmpHiRes  = 1./smpPerMsHiRes;
indSmp         = upK:upK:upK*numSmp;
%  ?PIX/SEC TO TRANSLATE AT SPECIFIED SPEED
pixDffPerMsL       = (spdDegPerSecL.*smpPerDeg)./1000;
pixDffPerMsR       = (spdDegPerSecR.*smpPerDeg)./1000;
%  ?PIX/SMP AT HI TEMPORAL SAMPLING RATE
pixDffPerSmpHiResL = pixDffPerMsL.*msPerSmpHiRes;
pixDffPerSmpHiResR = pixDffPerMsR.*msPerSmpHiRes;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CCD MOVIE AT HIGH TEMPORAL RESOLUTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PszXYbffrSq = size(LccdBffr,1)*[1 1];

[LccdBffrMV, LmovieCoords] = motionImagesCrop(LccdBffr, PszXYbffrSq, pixDffPerSmpHiResL, ...
  numSmpHiRes, [], 0, 'zeroDspTime', zeroDspTime);
[RccdBffrMV, RmovieCoords] = motionImagesCrop(RccdBffr, PszXYbffrSq, pixDffPerSmpHiResR, ...
  numSmpHiRes, [], 0, 'zeroDspTime', zeroDspTime);


if 0
    LccdBffrMV = zeros([size(LccdBffr), 1, numSmpHiRes]);
    RccdBffrMV = zeros([size(LccdBffr), 1, numSmpHiRes]);
    % TILT OF STIMULUS
    tiltDeg  = 90;
    % DISTANCE OF THE STIMULUS (FIX TO MONITOR DISTANCE)
    zp    = 1;
    % FUDGE TO EXAGGERATE PERSPECTIVE
    E = 1; % E = 100;
    % IppXm = zp.*tand(smpPos(smpPerDeg,PszXYbffr(1))).*E;
    % IppYm = zp.*tand(smpPos(smpPerDeg,PszXYbffr(2))).*E;
    IppXm = zp.*tand(smpPos(smpPerDeg,size(LccdBffr,2))).*E;
    IppYm = zp.*tand(smpPos(smpPerDeg,size(LccdBffr,1))).*E;
    [IppXm,IppYm]=meshgrid(IppXm,IppYm);

    upKtxt = 1;
    [IppXmTxt,IppYmTxt] = meshgrid(zp.*tand(smpPos(upKtxt.*smpPerDeg,size(LccdBffr,2))).*E,zp.*tand(smpPos(upKtxt.*smpPerDeg,size(LccdBffr,1))).*E);
    %%
    % CONVERT RETINAL SPEED TO PLANE MOTION
    [spdMtrPerSec tgtDirDeg]=retSpd2velocity(0,zp,spdDegPerSecL,spdDegPerSecR,IPDm,1);
    if tgtDirDeg == 180, sgnDir = 1; elseif tgtDirDeg == 0, sgnDir = -1; end
    mtrPerSmp = sgnDir.*spdMtrPerSec.*1e-3.*msPerSmpHiRes;
    zJitter = randInterval([-0.05 0.05],1,1);
    for i = 1:numSmp.*upK
        % PLOTTING IF NECESSARY
        if mod(i,upK) == 1, bPLOTindi = 1; else bPLOTindi =0 ; end
        % SLANT PLANE IF NECESSARY
        % [LccdBffr(:,:,:,i),RccdBffr(:,:,:,i)] = textureMapPlaneBinocular( LccdBffr,slntDeg,tiltDeg,zp+i.*mtrPerSmp,IppXm,IppYm,zp,+IPDm./2,IppXm,flipud(IppYm),bPLOTindi);
        zpSmp = zp+(i-1).*mtrPerSmp+zJitter;
        [LccdBffr(:,:,:,i),RccdBffr(:,:,:,i)] = textureMapPlaneBinocular(...
            LccdBffr, slntDeg, tiltDeg, zpSmp, IppXm, IppYm, zp, +IPDm./2, ...
            IppXmTxt, flipud(IppYmTxt), bPLOTindi);
        if bPLOTindi
            hold on;
            plotSquare([-diff(minmax(IppXm))./2 0], diff(minmax(IppYm))*[1 1]./2, 'y', 1);
            plotSquare([diff(minmax(IppXm))./2 0], diff(minmax(IppYm))*[1 1]./2, 'y', 1);
        end
    end
    % CROP TO BUFFER SIZE
    LccdBffr = cropImageCtr(LccdBffr,[],PszXYbffr);
    RccdBffr = cropImageCtr(RccdBffr,[],PszXYbffr);
    % REPLACE NANs WITH MEAN VALUE
    LccdBffr(isnan(LccdBffr)) = nanmean(LccdBffr(:));
    RccdBffr(isnan(RccdBffr)) = nanmean(RccdBffr(:));
    %%
end




%%%%%%%%%%%%%%%%%%%
% SLANT THE MOVIE % (IF DESIRED)
%%%%%%%%%%%%%%%%%%%
if slntDeg ~= 0
    for i = 1:5
        disp(['Iccd2IretS3D: WARNING! non-zero slants not yet handled!!! Write code!?']);
    end;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SENSOR TRANSFER FUNCTION % HI RES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('TIR','var') || isempty(TIR);
    TIR = coneTemporalImpulseResponse(msPerSmpHiRes.*[1:(numSmp.*upK)], 2, 0, 0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% OPTICS AND SENSOR TRANSFER FUNCTIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(lensInfo) && isstruct(lensInfo)
    PSF_Lbffr = padarray(lensInfo.PSF_L, PszRCdff/2);
    PSF_Rbffr = padarray(lensInfo.PSF_R, PszRCdff/2);
    OTF_Lbffr = fftshift(fft2(fftshift(PSF_Lbffr)));
    OTF_Rbffr = fftshift(fft2(fftshift(PSF_Rbffr)));
else
    if PSF == 1
        OTF_Lbffr = ones(PszRCbffr);
        OTF_Rbffr = ones(PszRCbffr);
    else
        error(['Iccd2IretS3D: WARNING! unhandled case']);
        PSF_Lbffr = padarray(PSF,PszRCdff/2);
        PSF_Rbffr = padarray(PSF,PszRCdff/2);
        OTF_Lbffr = fftshift(fft2(fftshift(PSF_Lbffr)));
        OTF_Rbffr = fftshift(fft2(fftshift(PSF_Rbffr)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY LENS OPTICS & TEMPORAL INTEGRATION FUNCTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE-TIME CONVOLUTION
LretBffr = convtemporal(LccdBffrMV, mean(LccdBffrMV(:)), TIR, OTF_Lbffr, 1, 0);
RretBffr = convtemporal(RccdBffrMV, mean(RccdBffrMV(:)), TIR, OTF_Rbffr, 1, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOWNSAMPLE TO DESIRED TEMPORAL RESOLUTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LccdBffrMV   = squeeze(LccdBffrMV(:,:,1,indSmp-upK+1));
RccdBffrMV   = squeeze(RccdBffrMV(:,:,1,indSmp-upK+1));
LretBffr   = squeeze(LretBffr(:,:,1,indSmp));
RretBffr   = squeeze(RretBffr(:,:,1,indSmp));

%%%%%%%%%%%%%%%%%%%%%%%
% CROP BUFFERED IMAGE %
%%%%%%%%%%%%%%%%%%%%%%%
Lccd = cropImageCtr(LccdBffrMV, [], PszXY);
Rccd = cropImageCtr(RccdBffrMV, [], PszXY);
Lret = cropImageCtr(LretBffr, [], PszXY);
Rret = cropImageCtr(RretBffr, [], PszXY);

%%%%%%%%%%%%%%%
% PRE-WINDOW? % % bake window into stimulus if desired
%%%%%%%%%%%%%%%
% Look with more detail how this windowing works, it may be off
if bPreWndw == 1
    % INPUT CONSISTENCY
    if sum(W(:))==numel(W) error(['Iccd2IretSPD: WARNING! bPreWndw=1 and W is all ones. No windowing will occur. Fix inputs!!!']); end
    % WEBER CONTRAST MOVIE %
    [LccdWeb, LccdDC] = contrastImage(Lccd,W);
    [RccdWeb, RccdDC] = contrastImage(Rccd,W);
    [LretWeb, LretDC] = contrastImage(Lret,W);
    [RretWeb, RretDC] = contrastImage(Rret,W);
    % WINDOW CONTRAST MOVIE
    LccdWeb = bsxfun(@times, LccdWeb, reshape(W, size(LccdWeb)));
    RccdWeb = bsxfun(@times, RccdWeb, reshape(W, size(RccdWeb)));
    LretWeb = bsxfun(@times, LretWeb, reshape(W, size(LretWeb)));
    RretWeb = bsxfun(@times, RretWeb, reshape(W, size(RretWeb)));
    % CONVERT BACK TO INTENSITY MOVIE
    Lccd = LccdWeb.*LccdDC + LccdDC;
    Rccd = RccdWeb.*RccdDC + RccdDC;
    Lret = LretWeb.*LretDC + LretDC;
    Rret = RretWeb.*RretDC + RretDC;
elseif bPreWndw == 0
    % WEBER CONTRAST MOVIE %
    [~, LccdDC] = contrastImage(Lccd,W);
    [~, RccdDC] = contrastImage(Rccd,W);
    [~, LretDC] = contrastImage(Lret,W);
    [~, RretDC] = contrastImage(Rret,W);
else error(['Iccd2IretSPD: WARNING! unhandled bPreWndw value. bPreWndw=' num2str(bPreWndw)]);
end

%[rmsContrast(Lccd,W,bPreWndw), rmsContrast(Lret,W,bPreWndw)];
%[rmsContrast(Rccd,W,bPreWndw), rmsContrast(Rret,W,bPreWndw)];
%%%%%%%%%%%%%%%%
% PLOT RESULTS %
%%%%%%%%%%%%%%%%
if bPLOT
    %%
    figure(12121);
    set(gcf,'position',[600 150 500 600]);
    for i = 1:size(LretBffr,length(size(LretBffr)))
        imagesc([Lret(:,:,i) Rret(:,:,i); Lccd(:,:,i) Rccd(:,:,i)]);
        axis image; axis xy;
        colormap gray;
        caxis(minmax([Lccd Rccd]));
        formatFigure('RET',[],'CCD')
        set(gca,'xtick',[]); set(gca,'ytick',[]);
        % pause(5.*msPerSmp./1000);
        pause(.25);
    end
end
