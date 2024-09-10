function LRSIpatchCL2BVXZ_variableDisp(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, ...
  dspStdArcMin, spdMeterPerSec, spdDirDeg, tgtPosZMeter, zeroDspTime, bWithLooming, projInfo, ...
  lensInfo, sensInfo, rndSdInfo, fdirBV)


% function LRSIpatchCL2BV(natORflt,numImg,stmPerLvlDTB,PszXYbffr,PszXY,dspArcMinAll,spdDegPerSecAll,bPreWndw,projInfo,lensInfo,sensInfo,wndwInfo,rndSdInfo,localORserver,bPLOT)
%
%   example call: LRSIpatchCL2BV('NAT',95,1000,[6 2]*[60 60],[60 60],0,[-8 0 8],projInfoStruct('S3D','bino','FLT',250),lensInfo,sensInfoStruct(60,60,250),wndwInfo,rndSdInfo,'both',1)
%
% creates and saves binocular movie structure (BV) from saved crop
% locations (CL), a specified lens model, and a specified sensor model
%
% natORflt:         type of depth structure to create
%                   'NAT' -> natural depth structure
%                   'FLT' -> flat    depth structure
% numImg:           number of images (e.g. 95)
% stmPerLvlDTB:     patches per stimulus level (e.g. 1000)
% PszXYbffr:        patch size with bffr (to prevent artifacts from blurring)
% PszXY:            patch size in pixels
% spdDegPerSecAll:  speed in the left eye's image
%                   right eye image speed is equal and opposite
% bPreWndw:         boolean to indicate whether to pre-window the stimulus
%                   1 -> bakes window into stimulus (good for psychophysics)
%                   0 -> does not                   (good for ...)
% projInfo:         project struct containing the following fields
% .durationMs:      movie duration
% .slntPriorType:   type of slant prior
%                   'FSP' -> flat   slant prior   (i.e. all surface frontoparllel)
%                   'CSP' -> cosine slant prior   (i.e. truncated cosine disttribution of slants)
%                   'USP' -> unifornm slant prior (i.e. all slants equally likely)
% lensInfo:         lens info struct (see lensInfoStruct.m )
% sensInfo          sensor info struct
% .spdDegPerSecAll: all speeds in deg per sec
% .smpPerDeg:       spatial resolution of movie
% .smpPerSec:       temporal resolution (frame-rate) of movie
% wndwInfo:         wndw info struct containing params... see wndwInfoStruct.m
% rndSdInfo         rndSd info struct containing params...
%                   rndSdCP:         random seed for sampling corresponding points
%                   rndSdGD:         random seed for sampling good patches
% localORserver:    location to save data to
%                   'server' -> server
%                   'local'  -> local
% bPLOT:            plot or not
%                   1 -> plot
%                   0 -> not
%%%%%%%%%%%%%%%%%%%%%%%%%
% SET PROJECT CODE
prjCode = 'S3D';

% UNPACK projInfo
smpPerDeg       = sensInfo.smpPerDeg;
smpPerSec       = sensInfo.smpPerSec;
durationMs      = projInfo.durationMs;

% Get IPD
IPD = LRSIcameraIPD();
% Set initial X position at 0
tgtPosX = 0;
% Set units to meters
units = 1;
% Duration of initial high resolution videos
smpPerSecHiRes = 480;
upK = ceil(smpPerSecHiRes/smpPerSec);

% GET THE COMBINATIONS OF SPEEDS AND DIRECTIONS. FIRST ROW IS SPEED, SECOND IS ANGLE
allMotions = customCombvec(spdMeterPerSec, spdDirDeg);
% Remove repeats of 0 speed motion
spd0Ind = find(allMotions(1,:)==0);
if (length(spd0Ind) > 1)
    allMotions(:,spd0Ind(2:end)) = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD CL STRUCTS: (C)rop (L)ocations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdirCLloc  = buildFolderNameDTB('LRSI', 'patch', prjCode, 'local');
fdirCLsrv  = buildFolderNameDTB('LRSI', 'patch', prjCode, 'server');
dspArcMin = 0; % ALWAYS LOAD IN ZERO DISPARITY STIMULI
fnameCL = buildFilenameLRSIpatchCL(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, ...
  dspArcMin, rndSdInfo);
CL = loadSavedStruct(fdirCLloc, fdirCLsrv, fnameCL, 'CL');

%% LOOP THROUGH directions/speeds of motion
for s = 1:size(allMotions, 2)
    tgtSpdMeter = allMotions(1,s);
    tgtDirDeg = allMotions(2,s);
    % Get the speed of each eye corresponding to this motion
    [spdDegPerSecL, spdDegPerSecR] = velocity2retSpd(tgtPosX, tgtPosZMeter, ...
      tgtSpdMeter, tgtDirDeg, IPD, units);

    % If speed is 0, make tgtDir NaN, since it doesn't make sense
    if tgtSpdMeter==0
        tgtDirDeg = NaN;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SKIP THIS SPEED IF AN BV STRUCT FOR THIS SPEED IS ALREADY AVAILABLE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BUILD FILE NAME FOR THIS SPEED
    fnameBV = buildFilenameLRSIpatchBV_XZ(natORflt, stmPerLvlDTB, PszXY, ...
        tgtSpdMeter, tgtDirDeg, tgtPosZMeter, dspStdArcMin);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LOOP OVER CROP LOCATIONS (CL) TO OBTAIN MONOCULAR IMAGES %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:stmPerLvlDTB
        %%%%%%%%%%%%%%%
        % LOAD IMAGES %
        %%%%%%%%%%%%%%%
        progressreport(i,5,stmPerLvlDTB)
        if i == 1 ||  CL.imgNumAll(i) ~=  CL.imgNumAll(i-1)
            % LOAD FULL LRSI IMAGES FROM DATABASE
            bInPaint = 0;
            [LphtFll, RphtFll] = loadLRSIimage(CL.imgNumAll(i), 1, bInPaint, 'PHT', 'img');
        end

         % MEMORY ALLOCATION
        if i == 1
            PszT = smpPerSec.*durationMs./1000;
            Lccd = zeros([fliplr(PszXY) PszT  stmPerLvlDTB]);
            Rccd = zeros([fliplr(PszXY) PszT  stmPerLvlDTB]);
            Lret = zeros([fliplr(PszXY) PszT  stmPerLvlDTB]);
            Rret = zeros([fliplr(PszXY) PszT  stmPerLvlDTB]);
            Lxyz = NaN; Rxyz = NaN;
            randomDspVec = zeros(stmPerLvlDTB,1);
        end

        % Sample a random offset for initial disparity
        randomDsp = randn()*dspStdArcMin;
        if abs(randomDsp)>30
            randomDsp = 30*sign(randomDsp);
        end
        randomDspVec(i) = randomDsp;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CROP IMAGE AS REQUIRED FOR NATURAL OR FLAT DEPTH VARIATION %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        interpType = 'linear';
        if strcmp(natORflt,'NAT')  % NATURAL IMAGES
            LphtBffr = cropImageCtrInterp(LphtFll, CL.LitpRC(i,:), PszXYbffr, interpType);
            RphtBffr = cropImageCtrInterp(RphtFll, CL.RitpRC(i,:), PszXYbffr, interpType);
        elseif strcmp(natORflt,'FLT')  % FLAT    IMAGES
            % Implement different sampling positions from eyes to match init disparity
            cropOffset = [0, randomDsp]; 
            if bWithLooming
                % Looming implements different init dsp, sample same patch from
                % left and right
                cropOffset = [0, 0]; 
            end
            if strcmp(CL.LorR(i),'L') % NOTE! Left eye images are repeated for flat structure
                LphtBffr = cropImageCtrInterp(LphtFll, ...
                    CL.LitpRC(i,:), PszXYbffr, interpType);
                RphtBffr = cropImageCtrInterp(LphtFll, ...
                    CL.RitpRC(i,:)+cropOffset, PszXYbffr, interpType);
            elseif strcmp(CL.LorR(i),'R') % NOTE! Left eye images are repeated for flat structure
                LphtBffr = cropImageCtrInterp(RphtFll, ...
                    CL.LitpRC(i,:)+cropOffset, PszXYbffr, interpType);
                RphtBffr = cropImageCtrInterp(RphtFll, ...
                    CL.RitpRC(i,:), PszXYbffr, interpType);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %
        % GENERATE BINOCULAR MOVIES: Mccd* AND Mret* %  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bPLOTmovie = 0;
        if ~bWithLooming
            [LccdMV, RccdMV] = Iccd2BVS3D(LphtBffr, RphtBffr, PszXY, spdDegPerSecL, ...
              spdDegPerSecR, smpPerDeg, smpPerSecHiRes, durationMs, zeroDspTime, bPLOTmovie);
        else
            dspDeg = randomDsp / 60; % Angle from target to eye, radians
            [LccdMV, RccdMV] = Iccd2BVLoomingS3D(LphtBffr, RphtBffr, PszXY, tgtSpdMeter, ...
              tgtDirDeg, tgtPosZMeter, smpPerDeg, smpPerSecHiRes, durationMs, ...
              zeroDspTime, dspDeg, IPD, bPLOTmovie);
        end
        [Lret(:,:,:,i), Rret(:,:,:,i), Lccd(:,:,:,i), Rccd(:,:,:,i)] = BVccd2BVret(LccdMV, ...
          RccdMV, durationMs, upK, lensInfo, bPLOTmovie);
        % PROGRESS REPORT
        progressreport(i,500,stmPerLvlDTB);
    end

    PszE = 2; % LRSIpatchCL2BV: WARNING! PszE hard coded to 2. b.c BV videos are binocular

    % CONTRAST IMAGE STRUCTURE
    BV = struct('CL', CL, 'natORflt', CL.natORflt, 'PszXYbffr', PszXYbffr, ...
                'PszXY', PszXY, 'PszE', PszE, 'PszT', PszT, ...
                'sensInfo', sensInfo, 'lensInfo', lensInfo, 'projInfo', projInfo, ...
                'tgtSpdMeter', tgtSpdMeter, 'tgtDirDeg', tgtDirDeg, ...
                'tgtPosZMeter', tgtPosZMeter, 'spdDegPerSecL', spdDegPerSecL, ...
                'spdDegPerSecR', spdDegPerSecR, 'stmPerLvlDTB', stmPerLvlDTB,...
                'smpPerDeg', smpPerDeg, 'smpPerSec', smpPerSec, 'durationMs', durationMs, ...
                'bWithLooming', bWithLooming, 'IPD', IPD, 'zeroDspTime', zeroDspTime, ...
                'smpPosDegX', sensInfo.smpPosDegX, 'smpPosDegY', sensInfo.smpPosDegY, ...
                'smpPosSecT', sensInfo.smpPosSecT, ...
                'sampleDisp', randomDspVec, ...
                'Lccd', Lccd, 'Rccd', Rccd, ...
                'Lret', Lret, 'Rret', Rret, ...
                'Lxyz', Lxyz, 'Rxyz', Rxyz, ...
                'fname', fnameBV);  % FILE NAME FOR SAVING

	% SAVE RESULTS IN STRUCTURE BV (INTENSITY MOVIE)
    %saveLRSIpatchPRJ(BV.fname, prjCode, localORserver, 1, BV, 'BV');
    BV = structDouble2Single(BV);
    S.BV = BV;
    save([fdirBV, BV.fname], '-struct', 'S', '-v7.3');
    
    disp(['LRSIpatchCL2BV : Intensity images now available for ', num2str(s), ' of ', ...
      num2str(size(allMotions, 2)), 'speeds']);
end
