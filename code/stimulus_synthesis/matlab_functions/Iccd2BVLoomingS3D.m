function [LccdMV, RccdMV] = Iccd2BVLoomingS3D(LccdBffr, RccdBffr, PszXY, ...
  tgtSpdMeter, tgtDirDeg, tgtPosZMeter, smpPerDeg, smpPerSec, durationMs, ...
  zeroDspTime, IPDm, bPLOT)


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
% GET THE SPATIAL TRANSLATION PER SAMPLE IN METERS
secPerSmp = 1/smpPerSec;  % How many s are in a sample
% If tgtSpdMeter==1, we may have tgtDirDeg is NaN
if tgtSpdMeter ~= 0
  xMtrPerSmp = tgtSpdMeter * secPerSmp * sind(tgtDirDeg);
  zMtrPerSmp = - tgtSpdMeter * secPerSmp * cosd(tgtDirDeg);
else 
  xMtrPerSmp = 0;
  zMtrPerSmp = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CCD MOVIE AT HIGH TEMPORAL RESOLUTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PszXYbffr = size(LccdBffr,1)*[1 1];

LccdMV = zeros([fliplr(PszXY) 1 numSmp]);
RccdMV = zeros([fliplr(PszXY) 1 numSmp]);

% TILT OF STIMULUS
tiltDeg  = 0;
slantDeg = 0;
% DISTANCE OF THE PROJECTION PLANE IN METERS
projPlZm = 1;
% FUDGE TO EXAGGERATE PERSPECTIVE
E = 1;

% Coordinates of projection plane in meters
projPlXm = projPlZm * tand( smpPos(smpPerDeg, PszXY(1))) * E;
projPlYm = projPlZm * tand( smpPos(smpPerDeg, PszXY(2))) * E;
[projPlXm, projPlYm] = meshgrid(projPlXm, projPlYm);

upKtxt = 1;

% Coordinates of texture in meters. Get texture size from angle
% and number of pixels
txtXm = tgtPosZMeter * tand(smpPos(smpPerDeg, size(LccdBffr,2)));
txtYm = tgtPosZMeter * tand(smpPos(smpPerDeg, size(LccdBffr,1)));
[txtXm, txtYm] = meshgrid(txtXm, txtYm);

% Allocate memory for the movie
LccdMV = zeros([fliplr(PszXY) 1 numSmp]);
RccdMV = zeros([fliplr(PszXY) 1 numSmp]);

for i = 1:numSmp
  txtZmSmp = tgtPosZMeter + (i-1) * zMtrPerSmp;
  txtXmSmp = txtXm + (i-1) * xMtrPerSmp;
  [LccdMV(:,:,:,i)] = textureMapPlane2(LccdBffr, txtZmSmp, ...
    txtXmSmp, txtYm, projPlZm, projPlXm, projPlYm, -IPDm/2, ...
    slantDeg, tiltDeg, 0);
  [RccdMV(:,:,:,i)] = textureMapPlane2(RccdBffr, txtZmSmp, ...
    txtXmSmp, txtYm, projPlZm, projPlXm, projPlYm, +IPDm/2, ...
    slantDeg, tiltDeg, 0);
end

% CROP TO BUFFER SIZE
%LccdBffr = cropImageCtr(LccdBffr,[],PszXYbffr);
%RccdBffr = cropImageCtr(RccdBffr,[],PszXYbffr);
%% REPLACE NANs WITH MEAN VALUE
%LccdBffr(isnan(LccdBffr)) = nanmean(LccdBffr(:));
%RccdBffr(isnan(RccdBffr)) = nanmean(RccdBffr(:));
%%

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
