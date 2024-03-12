function [I ctgIndSpd ctgIndDir XspdM XdirDeg S3D] = loadAMAdataS3D_XZ(spdMeterPerSec, spdDirDeg, ...
  trnORtst, localORserver, natORflt, numImg, stmPerLvlDTB, stmPerLvlFile, PszXY, ...
  projInfo, lensInfo, sensInfo, wndwInfo, bPreWndw, imgDim, dnK, rndSdInfo)

% function [I ctgInd X] = loadAMAdataS3D(spdDegPerSecAll,stmPerLvl,slntPriorType,trnORtst,localORserver,natORflt,numImg,stmPerLvlDTB,PszXY,projInfo,lensInfo,sensInfo,wndwInfo,bPreWndw,imgDim,dnK,rndSdInfo)
%
%   example call:
%
%
prjCode = 'S3D';

% UNPACK PROJECT INFO
smpPerDeg  = sensInfo.smpPerDeg;
smpPerSec  = sensInfo.smpPerSec;
durationMs = projInfo.durationMs;

% INPUT DEFAULTS
if ~exist('localORserver','var')  || isempty(localORserver) localORserver = 'local';   end;
if ~exist('natORflt', 'var')    || isempty(natORflt )  natORflt   = 'FLT'; end;
if ~exist('numImg', 'var')      || isempty(numImg )    numImg     = 095;       end;
if ~exist('stmPerLvlDTB','var') || isempty(stmPerLvlDTB)    stmPerLvlDTB    = 50;    end;
if ~exist('PszXY','var')        || isempty(PszXY)      PszXY      = [30 30];    end;
if ~exist('PszT','var')         || isempty(PszT)       PszT       = [15];    end;
if ~exist('imgDim','var')       || isempty(imgDim)     imgDim     = '1D'; end;
if ~exist('dnK'   ,'var')       || isempty(dnK)        dnK        = 2;    end;
if ~exist('smpPerDeg' ,'var')   || isempty(smpPerDeg)  smpPerDeg  = 30;    end;
if ~exist('smpPerSec' ,'var')   || isempty(smpPerSec)  smpPerSec  = 60;    end;
if ~exist('durationMs','var')   || isempty(durationMs) durationMs = 250;    end;
if ~exist('rndSdInfo','var')    || isempty(rndSdInfo) rndSdInfo = rndSdInfoStruct(1,1,1); end;

%%%%%%%%%%%%%%%%%%%%%%
% WINDOW INFO STRUCT % WINDOW PARAMETERS IN CALLING FUNCTION
%%%%%%%%%%%%%%%%%%%%%%
if ~exist('wndwInfo','var') || isempty(wndwInfo)
    wndwType = 'COS';
    bSymS = 0;            bSymT = 0;
    Xrmp = PszXY(1);      Xdsk = 0;
    Yrmp = PszXY(2);      Ydsk = 0;
    Trmp = ceil(PszT./2); Tdsk = floor(PszT./2);
    [wndwInfo] = wndwInfoStruct(wndwType,bSymS,bSymT,PszXY,PszT,Xrmp,Xdsk,Yrmp,Ydsk,Trmp,Tdsk);
end

%%%%%%%%%%%%%%%%%%%%
% LENS INFO STRUCT %
%%%%%%%%%%%%%%%%%%%%
if ~exist('lensInfo','var') || isempty(lensInfo)
    lensType     = {'NVR' 'NVR'};
    pupilMm      = 4;
    cplORidp     = 'IDP'; % OPTICS COUPLED TO DISPARITY VS INDEPENDENT OF DISPARITY
    natORfltLENS = 'FLT'; % OPTICS NATURAL (DEPTH CONSISTENT) OR FLAT
    lensInfo = lensInfoStruct(lensType,pupilMm,cplORidp,natORfltLENS,smpPerDeg,PszXY);
end

%%%%%%%%%%%%%%%%%%%
% BUILD FILE NAME %
%%%%%%%%%%%%%%%%%%%
fname = buildFilenameLRSIpatchS3D_XZ(natORflt, numImg, stmPerLvlDTB, PszXY, ...
  projInfo, lensInfo, sensInfo, wndwInfo, bPreWndw, imgDim, dnK, spdMeterPerSec, ...
  spdDirDeg, stmPerLvlFile, rndSdInfo, trnORtst);

%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD FOLDER NAME(S) %
%%%%%%%%%%%%%%%%%%%%%%%%
fdirLoc = buildFolderNameAMA('input','S3D','local');
fdirSrv = buildFolderNameAMA('input','S3D','server');

%%%%%%%%%%%%%
% LOAD DATA %
%%%%%%%%%%%%%
S3D = loadSavedStruct(fdirLoc, fdirSrv, fname, 'S3D');

I      = S3D.Iret;
ctgIndSpd = S3D.ctgIndSpd;
ctgIndDir = S3D.ctgIndDir;
XspdM = S3D.XspdM;
XdirDeg = S3D.XdirDeg;

