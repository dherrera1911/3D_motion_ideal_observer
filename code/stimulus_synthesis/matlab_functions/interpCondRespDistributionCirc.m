function [MUi, COVi, SIGMAi, KAPPAi, Xi] = interpCondRespDistributionCirc(X, MU, COV, SIGMA, KAPPA, ...
  interpFactor, method4interp, bPLOT, dims)

% function interpCondRespDistribution(AMA,MU,COV,SIGMA,KAPPA,method4interp,method4interpPos,interpFactor,bPLOT)
%
%   example call: % INTERPOLATED DISTRIBUTIONS WITH 8 TIMES THE SPACING OF AMA.X
%                   [MUi,COVi,~,~,Xi]=interpCondRespDistribution(X',MU,COV,[],[],8,'spline',1) 
%
%                 % INTERPOLATED DISTRIBUTIONS WITH FOUR TIMES THE SPACING OF AMA.X
%                   [MUi,COVi,SIGMAi,KAPPAi,Xi] = interpCondRespDistribution(PS.spdDegPerSec',MU,COV,[],[],32,'spline',0) ;
%
%                 % TO RESET AMA DISTRIBUTIONS
%                   for f = 1:AMA.numFilters, f, [AMA.MUi{f} AMA.COVi{f} AMA.SIGMAi{f} AMA.KAPPAi{f} AMA.Xi{f}] = interpCondRespDistribution(AMA.X',AMA.MU{f},AMA.COV{f},[],[],AMA.interpFactor,AMA.method4interp,0); end       
%
%                 % TEST DATA
%                   [MUtst  COVtst]=fitCondRespDistribution('gaussian',AMA.w,Itest,[],ctgIndTest,0.0,AMA.InormOpts,AMA.FrespOpts);        
%                   [MUtsti COVtsti,~,~, Xtsti] = interpCondRespDistribution(PS.spdDegPerSec',MUtst,COVtst,[],[],32,'spline',0);
%
%
% X:                     structure output from learnFiltersAMA
% MU:                    conditional likelihood mean vector       ('gaussian model' OR 'gengauss model')       
% COV:                   conditional likelihood covariance matrix ('gaussian model')      
% SIGMA:                 conditional likelihood scale             ('gengauss model')      
% KAPPA:                 conditional likelihood kappa             ('gengauss model')       
% interpFactor:          factor by which number of interpolated distributions 
%                        should be greater than originals 
% method4interp:         method used to interpolate distributions
%                       'spline' (preferred)
%                       'linear' (preferred)
%                       'pchip'  (anti-preferred)
% bPLOT:                 1 -> plot results
%                        0 -> don't
%%%%%%%%%%%%%%%%%%%%%
% MUi:                   interpolated mean matrix
% COVi:                  interpolated covariance matrix
% SIGMAi:                interpolated sigma matrix
% KAPPAi:                interpolated kappa matrix
% Xi:                    X values of interpolated distributions
% dPRIMEi:               interpolated dPrime distances between distributions

if ~exist('SIGMA','var') || isempty(SIGMA),           SIGMA = [];                    end
if ~exist('KAPPA','var') || isempty(KAPPA),           KAPPA = [];                    end
if ~exist('interpFactor','var')                       interpFactor = 4;              end
if ~exist('bPLOT','var'),                             bPLOT = 0;                     end
if ~exist('dims','var') || isempty(dims),             dims = 1:size(MU,2);           end

if any(~sort(X)==X)
    msg = 'Error: The X variable needs to be sorted';
    error(msg)
end

% USE ONLY SPECIFIED DIMS
X = X(:);
MU  = MU(:,dims);
COV = COV(dims, dims, :);

% SETUP INTERPOLATION OPTIONS % (number of Xs with which to densely interpolate) 
numSmpsTotal = length(X)*interpFactor+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTERPOLATE X, MEAN, COVARIANCE, POWER, AND SIGMA % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make it so we also interpolate between the first and last angles
Xinitial = min(X);
degDiff = min(mod(Xinitial-max(X),360), mod(Xinitial-max(X), 360));
Xfinal = max(X) + degDiff;
Xcirc = [X; Xfinal];
MUcirc = [MU; MU(1,:)];
COVcirc = COV;
COVcirc(:,:,size(COV,3)+1) = COVcirc(:,:,1);

Xi = linspace(Xinitial, Xfinal, numSmpsTotal);
% Remove last element, because it's repeated with the first one
Xi = Xi(1:(length(Xi)-1));

% INTERPOLATE MEAN VECTORS %
MUi = interp1(Xcirc', MUcirc, Xi', method4interp);

% INTERPOLATE COVARIANCE MATRICES
if size(COV,1) == 1
    COVi = reshape(interp1(Xcirc', squeeze(COVcirc), Xi', 'pchip'), [1 1 length(Xi)]);
else 
    % % INTERPOLATE COVARIANCE MATRICES
    % COVi = covInterp(X',MU,COV,method4interp,interpFactor,0);
    % INTERPOLATE COVARIANCE DIRECTLY 
    for m = 1:size(COV,1)
        for n = 1:size(COV,2)
            COVi(m,n,:) = interp1(Xcirc, squeeze(COVcirc(m,n,:)), Xi, method4interp);
        end
    end
end

% ERROR CHECKING
%%
for x = 1:length(Xi)
    %%
   [~, bNotPosSemDef] = chol(COVi(:,:,x));
   if bNotPosSemDef
      disp(['covInterp: WARNING! interpolated covar matrix #' num2str(x) ' is not positive semi-definite']);
      disp(['                    forcing positive semi-definite and setting bPLOT = 1']);
      [vec val]=eigs(COVi(:,:,x));
      val(val<0) = 0.01;
      COVi(:,:,x)=vec*val*vec';
   end
end

% INTERPOLATE SIGMA & KAPPA (gengauss only)
if ~isempty(KAPPA) && ~isempty(SIGMA)
    SIGMAi = covInterp(X',MU,SIGMA,method4interp,interpFactor,0);
    KAPPAi = interp1(X',KAPPA,Xi',method4interp);
else
    SIGMAi = []; 
    KAPPAi = [];
end

%%
if bPLOT
    disp(['interpCondRespDistribution: WARNING! rewrite plotting functions']);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PLOT COVARIANCE INTERPOLATION % (sanity check)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure('position',[861          15        1063        1069]);
    [nr,nc]=subplotRowsCols(size(COV,1).^2);
    if nr*nc <= 64
    for r = 1:nr
        for c = r:nc
            i = (r-1)*nc + c;
            subplot(nr,nc,i);
            hold on;
            plot(X,squeeze(COV(r,c,:))  ,'ko-' ,'linewidth',2);  
            plot(Xi,squeeze(COVi(r,c,:)),'k--','linewidth',2);
            if r == c
                formatFigure('X','Variance',[],0,0,18,14)
            else
                formatFigure('X','Covariance',[],0,0,18,14);
            end
        end
    end
    end
end
