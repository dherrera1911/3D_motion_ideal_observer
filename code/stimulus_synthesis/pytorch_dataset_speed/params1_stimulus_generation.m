%%%%%% Input the parameters for the data generation procedure

prjCode = 'S3D';
localORserver = 'local';
bSAVE = 1; % binary variable indicating whether to save

%%%%%%%%%%%%
%%%%% 0.1 MODIFIABLE DATABASE-RELATED PARAMETERS

% Datapoints in each latent variable level
stmPerLvlDTB = 800;
% Patch Size
PszXY = [60, 60]; % Size of stimuli to run through learning algorithm
% Sampling rate in time (equal monitor refresh if moving image to be presented)
smpPerSec = 60; 
% Movie duration in milliseconds
durationMs = 250;
% Standard deviation in arcmin of disparity variability
dspStdArcMin = 0;
% Starting distance from observer
tgtPosZMeter = 1;
% Set the magnitudes and directions of speeds to generate
maxSpd = 2.5;
spdStep = 0.1;
spdMeterPerSec = [0:spdStep:maxSpd];
%spdMeterPerSec = [spdStep/2:spdStep:maxSpd];
spdDirDeg = [-180, 0];
% Downsampling options
imgDim       = '1D';
dnK          = 2;
% Set the depth structure of the videos ('NAT' or 'FLT')
%   for 'FLT', instead of using the 2 original images, R is just
%   a duplicated and shifted version of L image
natORflt = 'FLT';

% Set whether the stimulus starts at 0 disparity, or
% if it has 0 disparity in the middle frame
zeroDspTime = 'start'; %alternative 'middle'

% Division of data between training and testing set
stmPerLvlTrn = 500; % 500; %1000
stmPerLvlTst = 300; %1000
spdNumLvlTrn = length(spdMeterPerSec); % 41;
spdNumLvlTst = 0;


% Plotting parameters
bPLOT_CP = 0;
bPLOTwndw = 0;
bPLOTbino = 0;

% Disparity for cl selection
%dspMinMax = [-15 15]; %[-22.5 22.5];  % 
%dspNumLvlTrn = 31; % 41; % 
%dspNumLvlTst = 31 % 41;
%dspNumLvl = 31; %41;
%dspArcMinTrn = linspace(dspMinMax(1), dspMinMax(2), dspNumLvlTrn);
%dspArcMinTst = linspace(dspMinMax(1), dspMinMax(2), dspNumLvlTst);
%dspArcMinAll = linspace(dspMinMax(1), dspMinMax(2), dspNumLvl);


%%%%%%%%%%%%
%%%%% 0.2 MODIFIABLE TECHNICAL DETAIL PARAMETERS

% Plot the corresponding points or not
% Choose images from database to use
imgNums = 1:98;
% Number of initial patches per LRSI image. Some will be discarded
stmPerImg = 30; 
% Size of patch to crop, with buffer to generate moving stim
PszXYbffr = [6 1].*[PszXY];
% Random Seeds
rndSdCP = 1; % rndSd for selecting potential corresponding points
rndSdGD = 1; % rndSd for selecting good      corresponding points
rndSdTT = 1; % rndSd for selecting trainng and test patches
rndSdInfo = rndSdInfoStruct(rndSdCP, rndSdGD, rndSdTT);
% Checks for bad corresponding points
chkDVNminDst = 4;
chkVRGmaxErr = 5;
chkNANmaxCnt = 0;
chkMINrmsCtr = 0.05;
% Take out bad images
imgNums(imgNums==60) = []; imgNums(imgNums==65) = []; imgNums(imgNums==85) = [];
numImg = length(imgNums);

% Slant prior type %%% NOT SURE WHERE TO PUT
slntPriorType = 'FSP';

%%%%%%%%%%%%
%%%%% 0.3 AUTOMATIC VALUES GENERATION, OR EQUIPMENT DEPENDENT

% Number of eyes
PszE = 2;
% Sampling rates for speed project
smpPerDeg = 60; % Pixels equaling 1 degree in original image acquisition
PszT = ceil(smpPerSec.*durationMs./1000);  % NUMBER OF FRAMES IN TIME
% Lens info struct
lensType     = {'NVR' 'NVR'};
pupilMm      = 4;
cplORidp     = 'IDP'; % Optics coupled to disparity vs independent of disparity
natORfltLENS = 'NAT'; % Optics natural (depth consistent) or flat
lensInfo     = lensInfoStruct(lensType, pupilMm, cplORidp, natORfltLENS, ...
  smpPerDeg, PszXY);

% Window info struct. Ensure window parametes are whole numbers
bPreWndw = 0;
wndwType = 'COS';  % 'NUN'          % WINDOW TYPE: bSym = 0 -> non-symmetric, bSym=1 -> symmetric
bSymS = 1;        bSymT = 0;        % SYMMETRY IN SPACE AND SPACE-TIME
Xrmp  = PszXY(1); Xdsk  = 0;
Yrmp  = PszXY(2); Ydsk  = 0;
Trmp  = PszT;     Tdsk  = 0;
[wndwInfo, W] = wndwInfoStruct(wndwType, bSymS, bSymT, PszXY, PszT, Xrmp, ...
  Xdsk, Yrmp, Ydsk, Trmp, Tdsk, bPLOTwndw);
% Project info data
monoORbino = 'bino';

projInfo = projInfoStruct(prjCode, monoORbino, slntPriorType, durationMs);
% Sensor info struct
sensInfo = sensInfoStruct(PszXY, smpPerDeg, smpPerSec, durationMs);

% Provisory until seeing what is the use of disparity:
dspMinMax = [0 0];
dspNumLvl = 1;

projectStruct = struct('projInfo', projInfo, ...
  'sensInfo', sensInfo, 'windowInfo', wndwInfo, ...
  'seedInfo', rndSdInfo);
  

