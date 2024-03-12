% MASTER WORKFLOW FOR
%   MODULE 0: PARAMETER SETTINGS FOR STIMULUS GENERATION AND SAVING
%   MODULE 3: APPLYING OPTICS AND SAVING BINOCULAR CCD AND RETINAL IMAGES
%   MODULE 4: CONVERTING BINOCULAR CCD AND RETINAL IMAGES TO AMA FORMAT


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODULE 0: PARAMETER SETTINGS FOR STIMULUS GENERATION AND SAVING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run("./params1_stimulus_generation.m");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODULE 3: CCD AND RETINAL IMAGES FOR ALL CROP LOCATIONS 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~strcmp(localORserver,'server')
    for i = 1:5
        disp(['LRSI2S3D2AMA: WARNING! call will save lotso data to local machine. Be careful!']);
    end
end

% Binocular images
bPLOTbino = 0;
bPreWndw = 0;

fdirBV = './generated_stim/BV_videos/';

dspArcMinAll = 0
%LRSIpatchCL2BVXZ(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, ...
%  dspArcMinAll, spdMeterPerSec, spdDirDeg, tgtPosZMeter, zeroDspTime, bPreWndw, ...
%  projInfo, lensInfo, sensInfo, wndwInfo, rndSdInfo, localORserver, bPLOTbino);

LRSIpatchCL2BVXZ_variableDisp(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, ...
  dspStdArcMin, spdMeterPerSec, spdDirDeg, tgtPosZMeter, zeroDspTime, bPreWndw, ...
  projInfo, lensInfo, sensInfo, wndwInfo, rndSdInfo, fdirBV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODULE 4 % CCD AND *RETINAL* IMAGES TO AMA FORMAT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHECK TO PREVENT STIMULUS OVERDRAFT
stmPerLvlTot = stmPerLvlTrn + stmPerLvlTst;
if stmPerLvlTot > stmPerLvlDTB,
   error(['LRSI2S3D2AMA: WARNING! More training and test stimuli requested than will be available! Rerun...']);
end

% GENERATE TRAINING SET
%stmPerLvl = stmPerLvlTrn;
indSmpStmExl = [];
trnORtst = 'TRN';
bPreWndw = 1;

[S3Dtrn, indSmpTrn] = LRSIpatchBV2S3DXZ(natORflt, numImg, stmPerLvlDTB, PszXY, ...
  dspStdArcMin, spdMeterPerSec, spdDirDeg, tgtPosZMeter, imgDim, dnK, ...
  stmPerLvlTrn, bPreWndw, projInfo, lensInfo, sensInfo, wndwInfo, trnORtst, ...
  indSmpStmExl, rndSdInfo, fdirBV);

fname = 'S3D-nStim_0500-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-TRN.mat'
fdir = './generated_stim/S3D_struct/';
save([fdir, fname], '-struct', 'S3Dtrn');


%% GENERATE TEST SET
if stmPerLvlTst>0
    [S3Dtst, indSmpTst] = LRSIpatchBV2S3DXZ(natORflt, numImg, stmPerLvlDTB, PszXY, ...
      dspStdArcMin, spdMeterPerSec, spdDirDeg, tgtPosZMeter, imgDim, dnK, ...
      stmPerLvlTst, bPreWndw, projInfo, lensInfo, sensInfo, wndwInfo, 'TST', ...
      indSmpTrn, rndSdInfo, fdirBV);
    fnameTst = 'S3D-nStim_0300-spdStep_0.100-maxSpd_2.50-dspStd_00-dnK_2-TST.mat'
    save([fdir, fnameTst], '-struct', 'S3Dtst');
end

