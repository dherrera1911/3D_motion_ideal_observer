% MASTER WORKFLOW FOR
%   MODULE 0: PARAMETER SETTINGS FOR STIMULUS GENERATION AND SAVING
%   MODULE 1: FINDING CORRESPONDING POINTS
%   MODULE 2: GETTING CROP LOCATIONS FOR ARBITRARY DISPARITIES
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
% MODULE 1: FIND POTENTIAL CORREPSONDING POINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CP = LRSIimage2CP(imgNums, stmPerImg, PszXYbffr, PszXY, chkDVNminDst, ...
  chkVRGmaxErr, chkNANmaxCnt, chkMINrmsCtr, rndSdInfo, bSAVE, localORserver, ...
  bPLOT_CP, 'wall');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODULE 2: SAMPLE CORRESPONDING POINTS & GET CROP LOCATIONS FOR NON-ZERO DISPARITIES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Provisory until seeing what is the use of disparity:
dspMinMax = [0 0];
dspNumLvl = 1;

LRSIpatchCP2CL(prjCode, numImg, stmPerImg, PszXYbffr, PszXY, stmPerLvlDTB, ...
  dspMinMax, dspNumLvl, natORflt, rndSdInfo, localORserver);

