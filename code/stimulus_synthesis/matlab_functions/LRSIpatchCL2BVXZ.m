function LRSIpatchCL2BVXZ(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, dspArcMinAll, ...
  spdMeterPerSec, spdDirDeg, tgtPosZMeter, zeroDspTime, bPreWndw, projInfo, lensInfo, sensInfo, ...
  wndwInfo, rndSdInfo, localORserver, bPLOT)

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

if length(dspArcMinAll)>1 error(['LRSIpatchCL2BV: WARNING! unhandled for length(dspArcMinAll) > 1']); end

% SET PROJECT CODE
prjCode = 'S3D';

% UNPACK projInfo
smpPerDeg       = sensInfo.smpPerDeg;
smpPerSec       = sensInfo.smpPerSec;
durationMs      = projInfo.durationMs;
slntPriorType   = projInfo.slntPriorType;

% Get IPD
IPD = LRSIcameraIPD();

% GET THE COMBINATIONS OF SPEEDS AND DIRECTIONS. FIRST ROW IS SPEED, SECOND IS ANGLE
allMotions = combvec(spdMeterPerSec, spdDirDeg);
% Remove repeats of 0 speed motion
spd0Ind = find(allMotions(1,:)==0);
if (length(spd0Ind) > 1)
    allMotions(:,spd0Ind(2:end)) = [];
end


% Set initial X position at 0
tgtPosX = 0;
% Set units to meters
units = 1;

% SLANT PRIOR: SAMPLE SLANTS TO SIMULATE
maxSlntDeg = 60;
slntDeg = slantPriorSamples(slntPriorType, maxSlntDeg, stmPerLvlDTB, 1, 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD CL STRUCTS: (C)rop (L)ocations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdirCLloc  = buildFolderNameDTB('LRSI', 'patch', prjCode, 'local');
fdirCLsrv  = buildFolderNameDTB('LRSI', 'patch', prjCode, 'server');
dspArcMin = 0; % ALWAYS LOAD IN ZERO DISPARITY STIMULI
fnameCL = buildFilenameLRSIpatchCL(natORflt, numImg, stmPerLvlDTB, PszXYbffr, PszXY, ...
  dspArcMin, rndSdInfo);
CL = loadSavedStruct(fdirCLloc, fdirCLsrv, fnameCL, 'CL');

%% LOOP THOUGH directions/speeds of motion
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
    % BUILD FILE DIRECTORY
    %fdirBV = buildFolderNameDTB('LRSI', 'patch', prjCode, localORserver);
    fdirBV = '/Users/Shared/VisionScience/Project_Databases/LRSI/Patch/MotionXZ';

    % BUILD FILE NAME FOR THIS SPEED
    fnameBV = buildFilenameLRSIpatchBV_XZ(natORflt, stmPerLvlDTB, PszXY, ...
                                        tgtSpdMeter, tgtDirDeg, tgtPosZMeter);

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
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CROP IMAGE AS REQUIRED FOR NATURAL OR FLAT DEPTH VARIATION %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        interpType = 'linear';
        if strcmp(natORflt,'NAT')  % NATURAL IMAGES
            LphtBffr = cropImageCtrInterp(LphtFll , CL.LitpRC(i,:), PszXYbffr, interpType);
            RphtBffr = cropImageCtrInterp(RphtFll , CL.RitpRC(i,:), PszXYbffr, interpType);
        elseif strcmp(natORflt,'FLT')  % FLAT    IMAGES
            if strcmp(CL.LorR(i),'L') % NOTE! Left eye images are repeated for flat structure
                LphtBffr = cropImageCtrInterp(LphtFll , CL.LitpRC(i,:), PszXYbffr, interpType);
                RphtBffr = cropImageCtrInterp(LphtFll , CL.RitpRC(i,:), PszXYbffr, interpType);
            elseif strcmp(CL.LorR(i),'R') % NOTE! Left eye images are repeated for flat structure
                LphtBffr = cropImageCtrInterp(RphtFll , CL.LitpRC(i,:), PszXYbffr, interpType);
                RphtBffr = cropImageCtrInterp(RphtFll , CL.RitpRC(i,:), PszXYbffr, interpType);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %
        % GENERATE BINOCULAR MOVIES: Mccd* AND Mret* %  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bPLOTmovie = 0;
        [Lret(:,:,:,i), Rret(:,:,:,i), Lccd(:,:,:,i), Rccd(:,:,:,i), LretDC(i,1), RretDC(i,1), ...
          LccdDC(i,1), RccdDC(i,1)] = Iccd2IretS3D_XZ(LphtBffr, RphtBffr, IPD, PszXY, ...
          spdDegPerSecL, spdDegPerSecR, smpPerDeg, smpPerSec, durationMs, zeroDspTime, lensInfo, ...
          [], [], wndwInfo, wndwInfo.W, bPreWndw, slntDeg(i), bPLOTmovie);

        %%%%%%%%%%%%%%%%%%%%
        % LOCAL STATISTICS %
        %%%%%%%%%%%%%%%%%%%%
        % RMS CONTRASTS
        LccdRMS(i,1) = rmsContrast(Lccd(:,:,:,i), wndwInfo.W, bPreWndw);
        RccdRMS(i,1) = rmsContrast(Rccd(:,:,:,i), wndwInfo.W, bPreWndw);
        LretRMS(i,1) = rmsContrast(Lret(:,:,:,i), wndwInfo.W, bPreWndw);
        RretRMS(i,1) = rmsContrast(Rret(:,:,:,i), wndwInfo.W, bPreWndw);

        %%%%%%%%%%%%%%
        %% PLOT STUFF % % (SANITY CHECK)
        %%%%%%%%%%%%%%
        if bPLOT
            figure(111); set(gcf,'position',[ 257 571 560 727]);
            for f = 1:size(Lret,3)
                subplot(3,2,1); cla; imagesc([LphtBffr].^.4);
                axis image;
                formatFigure([],[],['MphtL Bffr, i=' num2str(i)]);
                hold on;
                plotSquare(fliplr(size(LphtBffr)./2),PszXY,'y',1);
                cax = caxis;
                subplot(3,2,2); cla; imagesc([RphtBffr].^.4);
                axis image;
                formatFigure([],[],['MphtR Bffr, i=' num2str(i)]);
                hold on; plotSquare(fliplr(size(RphtBffr)./2),PszXY,'y',1); cax = caxis;
                subplot(3,1,2);
                imagesc([Lccd(:,:,f,i) Rccd(:,:,f,i)].^.4); axis image;
                formatFigure([],[],'MphtLR Ccd'); caxis([cax]);
                subplot(3,1,3);
                imagesc([Lret(:,:,f,i) Rret(:,:,f,i)].^.4); axis image;
                formatFigure([],[],'MphtLR Ret'); caxis([cax]);
                colormap gray; pause(.1);
            end
        end
        %%
        % PROGRESS REPORT
        progressreport(i,500,stmPerLvlDTB);
    end

    PszE = 2;
    disp(['LRSIpatchCL2BV: WARNING! PszE hard coded to 2. b.c BV videos are binocular']);

    % CONTRAST IMAGE STRUCTURE
    BV = struct('CL',CL,'natORflt',CL.natORflt,'PszXYbffr',PszXYbffr,'PszXY',PszXY, ...
                'PszE',PszE,'PszT',PszT,'bPreWndw',bPreWndw,'sensInfo',sensInfo, ...
                'wndwInfo',wndwInfo,'lensInfo',lensInfo,'projInfo',projInfo, ...
                'tgtSpdMeter',tgtSpdMeter,'tgtDirDeg',tgtDirDeg, ...
                'retSpdL', spdDegPerSecL, 'repSpdR', spdDegPerSecR, ...
                'tgtPosZMeter',tgtPosZMeter,'spdDegPerSecL',spdDegPerSecL, ...
                'spdDegPerSecR',spdDegPerSecR,'stmPerLvlDTB',stmPerLvlDTB,...
                'smpPerDeg',smpPerDeg,'smpPerSec',smpPerSec,'durationMs',durationMs, ...
                'smpPosDegX',sensInfo.smpPosDegX,'smpPosDegY',sensInfo.smpPosDegY, ...
                'smpPosSecT',sensInfo.smpPosSecT, ...
                'slntPriorType',slntPriorType,    ...
                'LccdDC',LccdDC,'RccdDC',RccdDC,  ...
                'LretDC',LretDC,'RretDC',RretDC,  ...
                'LccdRMS',LccdRMS,'RccdRMS',RccdRMS, ...
                'LretRMS',LretRMS,'RretRMS',RretRMS, ...
                'Lccd',Lccd,'Rccd',Rccd, ...
                'Lret',Lret,'Rret',Rret, ...
                'Lxyz',Lxyz,'Rxyz',Rxyz, ...
                'fname',fnameBV);                                                         % FILE NAME FOR SAVING

	% SAVE RESULTS IN STRUCTURE BV (INTENSITY MOVIE)
    saveLRSIpatchPRJ(BV.fname, prjCode, localORserver, 1, BV, 'BV');
    disp(['LRSIpatchCL2BV : Intensity images now available for ', num2str(s), ' of ', ...
      num2str(size(allMotions, 2)), 'speeds']);
end
