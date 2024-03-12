function [S3D, indSmpStm] = LRSIpatchBV2S3DXZ(natORflt, numImg, stmPerLvlDTB, PszXY, ...
  dspStdArcMin, spdMeterPerSec, spdDirDeg, tgtPosZMeter, imgDim, dnK, ...
  stmPerLvl, bPreWndw, projInfo, lensInfo, sensInfo, wndwInfo, trnORtst, ...
  indSmpStmExl, rndSdInfo, fdirBV)

% function [S3D,indSmpStm] = LRSIpatchBV2S3D(natORflt,numImg,stmPerLvlDTB,PszXY,projInfo,lensInfo,sensInfo,wndwInfo,bPreWndw,imgDim,dnK,spdArcMin,stmPerLvl,trnORtst,indSmpStmExl,localORserver,rndSdInfo)
%
% example call: % TRAINING SET
%                 [S3D,indSmpTrn] = LRSIpatchBV2S3D('NAT',95,1500,[60 60],lensInfo,wndwInfo,1,'1D',LRSIpixPerDeg(),'FLT',linspace(-16.875,16.875,19),stmPerLvl,'TRN',[],'local',rndSdInfo)
%               % TEST     SET
%                 [S3D,indSmpTst] = LRSIpatchBV2S3D(('NAT',95,1500,[60 60],lensInfo,wndwInfo,1,'1D',2,LRSIpixPerDeg(),'FLT',linspace(-16.875,16.875,145),stmPerLvl,'TST',indSmpTrn,'local',rndSdInfo)
%
% Extract a sample of patches from the LRSI binocular patch database with specified disparities & indices
%
%  REWRITE COMMENT LIST!!!
%
% Arguments for accessing saved B(inocular) I(mage) structures
%   natORflt      :  string indicating whether binocular patches will be sampled with or without depth structure
%                   'NAT' : patches with natural depth structure
%                   'FLT' : planar patches without natural depth structure
%   numImg        :  number of LRSI images from which the corresponding-points database is obtained
%   stmPerLvlDTB     :  number of patches per disparity in the saved BV structure
%   PszXY         :  patch-size in pixels
%   lensInfo      :  lens info structure
%   W             :  window
%   rndSdInfo:
% Arguments specifying sampling & annotation requirements
%  dnK           :  downsampling factor
%  spdArcMin  :  vector of disparities for which patches will be added to the output structure
%  indSmpStm        :  indices of binocular patches to be extracted from the saved BV structure(s)
%  trnORtst      :  string indicating whether the annotated image set saved will be used as a training or a test set
%                   'TRN' : Training set
%                   'TST' : Test set
% localORserver:    location to save data to
%                   'server' -> server
%                   'local'  -> local
% %%%%%%%%%%%%%%%% GENERATES %%%%%%%%%%%%%%%%%%%%%%%%%
% S3D
% .natORflt            :  string indicating whether binocular patches will be sampled with or without depth structure
%                      'NAT' : patches with natural depth structure
%                      'FLT' : planar patches without natural depth structure
% .trnORtst            :  string indicating whether the annotated image set saved will be used as a training or a test set
%                      'TRN' : Training set
%                      'TST' : Test set
% .indTrn/.indTst      : indices of binocular patches extracted from the saved BV structure(s)
% .CL                  : structure with data on crop-locations of the saved binocular patches
% .BVfname             : names of parent BV structures
% .Xtrn/Xtst           : vector of disparities represented (equally i.e. stmPerLvl each) in the output database
% .ctgIndTrn/ctgIndTst : category indiex
% .Iccd                : binocular ccd image patches
% .Iret                : binocular retinal image patches

% PROJECT CODE
prjCode = 'S3D';

% DATA HANDLING OF TRN/TST
if ~strcmp(trnORtst,'TRN') && ~strcmp(trnORtst,'TST')
    error('LRSIpatchBV2S3D: Warning! Unhandled value of trnORtst. Set to TRN or TST and rerun.');
end

% INSIST ON PASSING EXCLUDED INDICES FOR TEST SET GENERATION
if strcmp(trnORtst,'TST')
    if ~exist('indSmpStmExl','var') || isempty(indSmpStmExl)
        error('LRSIpatchBV2S3D: Warning! Training set indices missing. They are need to generate a non-overlapping test set. Specify and rerun.');
    end
end

% MAKE SURE TRAINING/TEST SET SIZE IS IN ALLOWABLE LIMITS
if stmPerLvl > (stmPerLvlDTB - numel(indSmpStmExl))
    error('LRSIpatchBV2S3D: Warning! Stimulus overdraft! More stimuli per level are sought than are available');
end

dspArcMin = 0; %Hard coded, mean 0 disparity

% UNPACK projInfo
durationMs    = projInfo.durationMs;

% UNPACK sensInfo STRUCT
smpPerDeg     = sensInfo.smpPerDeg;
smpPerSec     = sensInfo.smpPerSec;

% GENERATE VECTOR OF MOTIONS TO PROCESS
allMotions = combvec(spdMeterPerSec, spdDirDeg);
% Remove repeats of 0 speed motion
spd0Ind = find(allMotions(1,:)==0);
if (length(spd0Ind) > 1)
    allMotions(:,spd0Ind(2:end)) = [];
end

% SAMPLE POSITIONS IN SPACE AND TIME
smpPosDegX = smpPosDownsample(sensInfo.smpPosDegX, dnK);
smpPosDegY = smpPosDownsample(sensInfo.smpPosDegY, dnK);
smpPosSecT = sensInfo.smpPosSecT;

% AVERAGE VERTICALLY (OR NOT)
if strcmp(imgDim,'2D')
  % DO NOTHING
elseif strcmp(imgDim,'1D')
  smpPosDegY = 0;
end

% SIZE OF DOWNSAMPLED POTENTIALLY AVERAGED 'IMAGE' (IMPORTANT!!!)
Isz = [length(smpPosDegY), length(smpPosDegX), length(smpPosSecT)];

%IF TEST SET IS BEING GENERATED THEN EXCLUDE TRAINING SET INDICES FROM DRAW
indSmpAll = [1:stmPerLvlDTB]';
if strcmp(trnORtst, 'TRN')
    indSmpFree = indSmpAll;
elseif strcmp(trnORtst, 'TST')
    indSmpFree = setdiff(indSmpAll, indSmpStmExl);
end
% SET RANDOM SEED
setRndSd(rndSdInfo.rndSdTT);

% DRAW SAMPLES FOR DATA-SET
indSmpStm = randsample(indSmpFree, stmPerLvl, false);
indSmpStm = sort(indSmpStm);

% NECESSARY FOR USE WITH CROP LOCATIONS FOR MANY DISPARITIES
CL = [];

% PREALLOCATE MEMORY
Iccd = zeros(2.*prod(Isz), stmPerLvl.*size(allMotions,2));
Iret = zeros(2.*prod(Isz), stmPerLvl.*size(allMotions,2));
BVfname = {};

spdDegPerSecL = [];
sampleDspVec = [];
% CONCATENATE IMAGE VECTORS
for d = 1:size(allMotions,2)
    tgtSpdMeter = allMotions(1,d);
    tgtDirDeg = allMotions(2,d);

    % If speed is 0, make tgtDir NaN, since it doesn't make sense
    if tgtSpdMeter==0
        tgtDirDeg = NaN;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LOAD BV STRUCTS: (B)inocular (I)images %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BUILD FILE DIRECTORY
%    fdirBV = buildFolderNameDTB('LRSI', 'patch', prjCode, localORserver);
    % BUILD FILE NAME FOR THIS SPEED
    fnameBV = buildFilenameLRSIpatchBV_XZ(natORflt, stmPerLvlDTB, PszXY, tgtSpdMeter, ...
      tgtDirDeg, tgtPosZMeter, dspStdArcMin);
    BV = loadSavedStruct(fdirBV, fdirBV, fnameBV, 'BV');

    BV = structElementSelect(BV, indSmpStm, stmPerLvlDTB);
    BVfname{d} = BV.fname; %%% check that this works as intended

    CLdsp = structElementSelect(BV.CL, indSmpStm, stmPerLvlDTB);
    CLdsp = rmfield(CLdsp, {'fnameCP','fnameDSP'});
    CL = structmerge(CL, CLdsp);

    % WINDOW IF NECESSARY
    if bPreWndw
        for s = 1:stmPerLvl
            BV.Lccd(:,:,:,s) = contrastWindowing(BV.Lccd(:,:,:,s), wndwInfo.W);
            BV.Rccd(:,:,:,s) = contrastWindowing(BV.Rccd(:,:,:,s), wndwInfo.W);
            BV.Lret(:,:,:,s) = contrastWindowing(BV.Lret(:,:,:,s), wndwInfo.W);
            BV.Rret(:,:,:,s) = contrastWindowing(BV.Rret(:,:,:,s), wndwInfo.W);
        end
    end

    % DOWNSAMPLE IF NECESSARY
    Lccd = imresize(BV.Lccd, 1/dnK, 'bilinear');
    Rccd = imresize(BV.Rccd, 1/dnK, 'bilinear');
    Lret = imresize(BV.Lret, 1/dnK, 'bilinear');
    Rret = imresize(BV.Rret, 1/dnK, 'bilinear');

    % VERTICAL AVERAGE OR NOT
    if strcmp(imgDim,'1D')
      Lccd = mean(Lccd,1);
      Rccd = mean(Rccd,1);
      Lret = mean(Lret,1);
      Rret = mean(Rret,1);
    else % DO NOTHING
    end

    % CONCATENATES L/R INTO COLUMN VECTOR
    IccdDsp = cat(1,reshape(Lccd,[],size(Lccd,length(size(Lccd)))), ...
                    reshape(Rccd,[],size(Rccd,length(size(Rccd)))));
    IretDsp = cat(1,reshape(Lret,[],size(Lret,length(size(Lret)))), ...
                    reshape(Rret,[],size(Lret,length(size(Lret)))));

    % STIMULUS INDICES BY SPEED
    indSpd = (d-1).*stmPerLvl + [1:stmPerLvl]';

    % Save the vector of stimulus disparities
    if isfield(BV, "sampleDisp")
        sampleDisp = BV.sampleDisp;
    else
        sampleDisp = zeros(stmPerLvl,1);
    end
    sampleDspVec = [sampleDspVec; BV.sampleDisp];

    % STORE DATA IN Iccd & Iret (INTENSITY IMAGE)
    Iccd(:,indSpd) = IccdDsp;
    Iret(:,indSpd) = IretDsp;

    % Grow the vector of retinal speeds, to keep in the struct for reference
    spdDegPerSecL(length(spdDegPerSecL)+1) = BV.spdDegPerSecL;

    progressreport(d,5,size(allMotions,2));
end

% RECORD GROUND-TRUTH LABELS
Xspd = spdMeterPerSec;
Xdir = spdDirDeg;
Xmotion = allMotions;
Xmotion(2,Xmotion(1,:)==0) = NaN;
XspdRet = spdDegPerSecL;

[~, XspdInd] = ismember(Xmotion(1,:), Xspd);
XspdInd = repmat(XspdInd, stmPerLvl, 1);
XspdInd = XspdInd(:);

[~, XdirInd] = ismember(Xmotion(2,:), Xdir);
XdirInd = repmat(XdirInd, stmPerLvl, 1);
XdirInd = XdirInd(:);

XmotionInd = repmat(1:size(Xmotion,2), stmPerLvl, 1);
XmotionInd = XmotionInd(:);

[XspdRet, retMotionOrder] = sort(XspdRet);
XspdRetInd = repmat(retMotionOrder, stmPerLvl, 1);
XspdRetInd = XspdRetInd(:); 

DSP       = dspArcMin;

dspStd = round(std(sampleDspVec));

% smpPerSec = 0;
% durationMs = 0;

% OBTAIN WINDOW OF THE SAME SIZE AS THE SAVED STIMULI
WdnK = wndwInfo.W;
Wrsz = imresize(wndwInfo.W,1/dnK,'bilinear'); % THIS ASSUMES THAT THE WINDOW STRUCTURE HAD WINDOWS ORIGINALLY OF THE SIZE OF Lccd
if numel(wndwInfo.W) ~= size(Iret,1)
    WdnK = Wrsz;
end

if strcmp(imgDim,'1D')
    %VERTICALLY AVERAGE
    WdnK = mean(WdnK, 1);
end

% RESHAPE WINDOW FOR SAVING
W = [WdnK WdnK];  %WINDOW FOR CONCATENATED BVNOCULAR IMAGE
W = reshape(W,[],1);

% GENERATE FILE-NAME FOR SAVING
fname = buildFilenameLRSIpatchS3D_XZ(natORflt, numImg, stmPerLvlDTB, PszXY, ...
  projInfo, lensInfo, sensInfo, wndwInfo, bPreWndw, imgDim, dnK, spdMeterPerSec, ...
  spdDirDeg, stmPerLvl, rndSdInfo, trnORtst);

% GENERATE OUTPUT STRUCTURE (BVNOCULAR PATCHES WITH GROUND-TRUTH DISPARITIES)
S3D = struct('prjCode', prjCode, 'natORflt', natORflt, ...
             'trnORtst', trnORtst, ... %%'slntPriorType', BV.slntPriorType, ...
             'indSmpAll', indSmpAll, 'indSmpStm', indSmpStm, 'BVfname', {BVfname}, 'CL', CL, ...
             'XspdM', Xspd, 'XdirDeg', Xdir, 'Xmotion', Xmotion, 'XspdRet', XspdRet, ...
             'DSP', dspArcMin, 'dspStdArcMin', dspStdArcMin, ...
             'ctgIndSpd', XspdInd, 'ctgIndDir', XdirInd, ...
             'ctgIndMotion', XmotionInd, 'ctgIndRetSpd', XspdRetInd, ...
             'stmPerLvl', length(indSmpStm), ...
             'Isz', Isz, 'monoORbino', projInfo.monoORbino, ...
             'Iccd', Iccd, 'Iret', Iret, ...
             'imgDim', imgDim, 'dnK', dnK, ...
             'smpPerDeg', BV.smpPerDeg./dnK, 'smpPerSec', BV.smpPerSec, 'durationMs', BV.durationMs, ...
             'smpPosDegX', smpPosDegX, 'smpPosDegY', smpPosDegY, 'smpPosSecT', smpPosSecT, ...
             'sampleDisp', sampleDspVec, ...
             'projInfo', BV.projInfo, ...
             'lensInfo', BV.lensInfo, ...
             'sensInfo', BV.sensInfo, ...
             'wndwInfo', wndwInfo, ...
             'W', W, ...
             'bPreWndw', bPreWndw, ...
             'rndSdInfo', rndSdInfo,...
             'fname', fname);

% SAVE OUTPUT STRUCTURE
%saveAMAdataPRJ(S3D.fname, 'input', prjCode, localORserver, 1, S3D, prjCode);

