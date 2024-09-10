function [S3D] = makeGratingBVAMA(stmPerLvl, PszXY, spdMeterPerSec, spdDirDeg, tgtPosZMeter, durationMs, lensInfo, wndwInfo, dnK)

%%%%%% Some parameters
% Sampling resolution parameters
smpPerDeg = 60;
smpPerSec = 60;
% Target initial position in X axis
tgtPosX = 0;
% Set units to meters
units = 1;
%
slntDeg = 0;
% Get IPD and other parameters
IPD = LRSIcameraIPD();
%
zeroDspTime = 'start';

%%%%% Make Buffer grating images
xBffSizePix = PszXY(1)*10; % x size Buffer
yBffSizePix = PszXY(2); % y size buffer
% Get horizontal and vertical position in visual degrees
xBffr = smpPos(smpPerDeg, xBffSizePix);
yBffr = smpPos(smpPerDeg, yBffSizePix);
% grating parameters
cyclePerDeg = [0.5, 1, 2, 4];
A = 1;
% make grating images
for i = 1:stmPerLvl
    phaseDeg = randi(360, 1, 4);
    gratingBffr{i} = makeVerticalCompoundGrating(xBffr, yBffr, cyclePerDeg, A, phaseDeg);
    gratingBffr{i} = round((mat2gray(gratingBffr{i})*(256^2))); % take to 1-256^2 range
end

% GET THE COMBINATIONS OF SPEEDS AND DIRECTIONS. FIRST ROW IS SPEED, SECOND IS ANGLE
allMotions = customCombvec(spdMeterPerSec, spdDirDeg);
% Remove repeats of 0 speed motion
spd0Ind = find(allMotions(1,:)==0);
if (length(spd0Ind) > 1)
    allMotions(:,spd0Ind(2:end)) = [];
end

% Get the downsampled patch coordinates in visual field degrees
x = smpPos(smpPerDeg, PszXY(1));
y = smpPos(smpPerDeg, PszXY(2));
smpPosDegX = smpPosDownsample(x, dnK);
smpPosDegY = 0; % 0 because we average vertically
smpPosSecT = smpPos(smpPerSec, smpPerSec.*durationMs./1000,1)' + 1./smpPerSec;
Isz = [length(smpPosDegY), length(smpPosDegX), length(smpPosSecT)];

% start some variables to fill with the data
Iccd = zeros(2.*prod(Isz), stmPerLvl.*size(allMotions,2));
Iret = zeros(2.*prod(Isz), stmPerLvl.*size(allMotions,2));
spdDegPerSecLVec = [];

for s = 1:size(allMotions, 2)
    % Get the speed of each eye corresponding to this motion
    tgtSpdMeter = allMotions(1,s);
    tgtDirDeg = allMotions(2,s);
    [spdDegPerSecL, spdDegPerSecR] = velocity2retSpd(tgtPosX, tgtPosZMeter, ...
      tgtSpdMeter, tgtDirDeg, IPD, units);

    % If speed is 0, make tgtDir NaN, since it doesn't make sense
    if tgtSpdMeter==0
        tgtDirDeg = NaN;
    end

    % Generate the original binocular movies
    bPLOTmovie = 0;
    Lccd = [];
    Rccd = [];
    Lret = [];
    Rret = [];
    for i = 1:stmPerLvl
        bPreWndw = 0;
        [Lret(:,:,:,i), Rret(:,:,:,i), Lccd(:,:,:,i), Rccd(:,:,:,i)] = ...
          Iccd2IretS3D_XZ(gratingBffr{i}, gratingBffr{i}, IPD, PszXY, ...
          spdDegPerSecL, spdDegPerSecR, smpPerDeg, smpPerSec, durationMs, zeroDspTime, ...
          lensInfo, [], [], wndwInfo, wndwInfo.W, bPreWndw, slntDeg, bPLOTmovie);
        % Apply window to the generated stimuli 
        Lccd(:,:,:,i) = contrastWindowing(Lccd(:,:,:,i), wndwInfo.W);
        Rccd(:,:,:,i) = contrastWindowing(Rccd(:,:,:,i), wndwInfo.W);
        Lret(:,:,:,i) = contrastWindowing(Lret(:,:,:,i), wndwInfo.W);
        Rret(:,:,:,i) = contrastWindowing(Rret(:,:,:,i), wndwInfo.W);
    end

    % DOWNSAMPLE VIDEOS
    LccdDwn = imresize(Lccd, 1/dnK, 'bilinear');
    RccdDwn = imresize(Rccd, 1/dnK, 'bilinear');
    LretDwn = imresize(Lret, 1/dnK, 'bilinear');
    RretDwn = imresize(Rret, 1/dnK, 'bilinear');

    % VERTICAL AVERAGE
    LccdAvg = mean(LccdDwn,1);
    RccdAvg = mean(RccdDwn,1);
    LretAvg = mean(LretDwn,1);
    RretAvg = mean(RretDwn,1);

    % CONCATENATES L/R INTO COLUMN VECTOR
    IccdBV = cat(1,reshape(LccdAvg,[],size(LccdAvg,length(size(LccdAvg)))), ...
                    reshape(RccdAvg,[],size(RccdAvg,length(size(RccdAvg)))));
    IretBV = cat(1,reshape(LretAvg,[],size(LretAvg,length(size(LretAvg)))), ...
                    reshape(RretAvg,[],size(LretAvg,length(size(LretAvg)))));

    % STIMULUS INDICES BY SPEED
    indSpd = (s-1).*stmPerLvl + [1:stmPerLvl]';

    % STORE DATA IN Iccd & Iret (INTENSITY IMAGE)
    Iccd(:,indSpd) = IccdBV;
    Iret(:,indSpd) = IretBV;

    % STORE INDEX VECTORS
    [~, lvlSpdInd] = ismember(tgtSpdMeter, spdMeterPerSec);
    [~, lvlDirInd] = ismember(tgtDirDeg, spdDirDeg);
    
    XspdInd(indSpd) = lvlSpdInd;
    XdirInd(indSpd) = lvlDirInd;
    XmotionInd = s;
    spdDegPerSecLVec = [spdDegPerSecLVec, spdDegPerSecL];
end

Xspd = spdMeterPerSec;
Xdir = spdDirDeg;
Xmotion = allMotions;
Xmotion(2,Xmotion(1,:)==0) = NaN;
XspdRet = spdDegPerSecL;

% Make the X variable and index for retinal speed
[XspdRet, retMotionOrder] = sort(spdDegPerSecLVec);
XspdRetInd = repmat(retMotionOrder, stmPerLvl, 1);
XspdRetInd = XspdRetInd(:); 

% OBTAIN WINDOW OF THE SAME SIZE AS THE SAVED STIMULI
WdnK = wndwInfo.W;
Wrsz = imresize(wndwInfo.W,1/dnK,'bilinear'); % THIS ASSUMES THAT THE WINDOW STRUCTURE HAD WINDOWS ORIGINALLY OF THE SIZE OF Lccd
if numel(wndwInfo.W) ~= size(Iret,1)
    WdnK = Wrsz;
end

%VERTICALLY AVERAGE Window Window
WdnK = mean(WdnK, 1);

% RESHAPE WINDOW FOR SAVING
W = [WdnK WdnK];  %WINDOW FOR CONCATENATED BVNOCULAR IMAGE
W = reshape(W,[],1);

% GENERATE OUTPUT STRUCTURE (BVNOCULAR PATCHES WITH GROUND-TRUTH DISPARITIES)
S3D = struct('prjCode', 'S3DXY', 'XspdM', Xspd, 'XdirDeg', Xdir, 'Xmotion', Xmotion, ...
             'XspdRet', XspdRet, 'ctgIndSpd', XspdInd, 'ctgIndDir', XdirInd, ...
             'ctgIndMotion', XmotionInd, 'ctgIndRetSpd', XspdRetInd, ...
             'stmPerLvl', stmPerLvl, 'Isz', Isz, 'Iccd', Iccd, 'Iret', Iret, ...
             'imgDim', '1D', 'dnK', dnK, 'smpPerDeg', smpPerDeg/dnK, ...
             'smpPerSec', smpPerSec, 'durationMs', durationMs, ...
             'smpPosDegX', smpPosDegX, 'smpPosDegY', smpPosDegY, ...
             'smpPosSecT', smpPosSecT, 'lensInfo', lensInfo, 'wndwInfo', ...
             wndwInfo, 'W', W, 'isWindowed', 1);

