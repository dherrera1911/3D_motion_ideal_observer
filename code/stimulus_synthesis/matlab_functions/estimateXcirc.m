function [XHATi, PPi, CLi, Xi, MUi, COVi, SIGMAi, KAPPAi] = estimateXcirc(modelCR, ...
  X, sTrn, fTrn, rTrn, ctgIndTrn, sTst, fTst, rTst, ctgIndTst, rMax, fano, var0, ...
  method4est, meanORmedian, method4interp, interpFactor, prior, bPLOT, CI)

% function [XHATi, PPi, CLi, Xi, MUi, COVi, SIGMAi, KAPPAi] = estimateXnew(modelCR, X, sTrn, ...
%   fTrn, rTrn, ctgIndTrn, sTst, fTst, rTst, ctgIndTst, rMax, fano, var0, method4est, ...
%   meanORmedian, method4interp, interpFactor, prior, bPLOT, CI)   
%                                                                              
%   example call: [s ctgInd     X] = loadAMAdevData('Disparity','Sml');
%                 [f E minTimeSec] = amaR01('GSS','MAP',2,0,1,[],s,ctgInd,X,1,.5,0.05,[]);
%                 [XHATi,PPi,CLi,Xi,MUi,COVi,SIGMAi,KAPPAi] = estimateXnew('gaussian',X,s,f,[],ctgInd,[],[],[],[],rMax,fano,var0,'MAP','median','spline',8,[],1); )
%
% modelCR:       'gaussian' -> gaussian fit to conditional response dstb
%                'gengauss' -> generalized gaussian fit to CL
% X:             latent variable values                  [  1   x nLvl ]
% sTrn:          stimulus          (training)            [ nDim x nStm ]
% fTrn:          filter weights    (training)            [ nDim x  nF  ]
% rTrn:          responses         (training)            [ nDim x  nF  ]
% ctgIndTrn:     category indices  (training)            [ nStm x  1   ]
% sTst:          stimulus          (  test  )            [ nDim x nStm ]
% fTst:          filter weights    (  test  )            [ nDim x  nF  ]
% rTst:          responses         (  test  )            [ nDim x  nF  ]
% ctgIndTst:     category indices  (  test  )            [ nStm x  1   ]
% rMax:          response max
% fano:          fano factor 
% var0:          baseline variance
% method4est:    method for reading out the posterior probability
%                 'MMSE' -> minium mean squared error
%                 'MAP'  -> maximum a posteriori estimate
% meanORmedian:   'mean'   plots the mean estimate
%                 'median' plots the median estimate
% method4interp: method used to interpolate distributions
%                'spline' (preferred)
%                'linear' (preferred) 
%                'pchip'  (anti-preferred)
% interpFactor:  factor by which number of interpolated distributions 
%                should be greater than original  
% prior:            prior                                   [   1 x numCtg ]
% bPLOT:         plot or not
%                1 -> plot
%                0 -> not
% CI:            confidence interval
%%%%%%%%%%%%%%%%%%%%%
% XHATi:      X estimate based on the posterior probability distribution
% Xi:                    X values of interpolated distributions
% CL:         likelihood                               [nStm x nCtg]
% PPi:        posterior probability pf X               [nStm x nCtg]
% MU:         mean matrix        (    gaussian & gengauss model      )
%                                (               gengauss model      )
% COV:        covariance matrix  (    gaussian            model      )
%                                         
% SIGMA:      scale matrix       (               gengauss model      ) 
%                                ( ... see gengaussSigma2Cov.m   ... )
% KAPPA:      kappa matrix       (               gengauss model      ) 
%                                ( ... see gengaussKappa2Kurtosis.m  )

% COMPUTE/READ-IN RESPONSES
if ~exist('rTrn',     'var') || isempty(rTrn)      rTrn      = stim2resp(sTrn,fTrn,rMax); 
else disp(['estimateXnew: WARNING! using user input rTrn, not fTrn and sTrn for responses']); end
if ~exist('sTst',     'var') || isempty(sTst)      sTst      = sTrn;                          end
if ~exist('fTst',     'var') || isempty(fTst)      fTst      = fTrn;                          end
if ~exist('rTst',     'var') || isempty(rTst)      rTst      = stim2resp(sTst,fTst,rMax);     
else disp(['estimateXnew: WARNING! using user input rTst, not fTst and sTst for responses']); end
if ~exist('ctgIndTst','var') || isempty(ctgIndTst) ctgIndTst = ctgIndTrn;                     end
if ~exist('prior','var')        || isempty(prior)        prior        = [];                            end
if ~exist('CI'       ,'var') || isempty(CI)        CI        = [];                            end

% FIT CONDITIONAL RESPONSE DISTRIBUTIONS (FROM TRAINING STIMULI)
[MU, COV, SIGMA, KAPPA] = fitCondRespDistribution(modelCR, sTrn, fTrn, ...
  rTrn, ctgIndTrn, rMax);

% INTERPOLATE CONDITIONAL RESPONSE DISTRIBUTIONS 
[MUi, COVi, SIGMAi, KAPPAi, Xi] = interpCondRespDistributionCirc(X, MU, ...
  COV, SIGMA, KAPPA, interpFactor, method4interp, 0);

% COMPUTE LIKELIHOOD
CLi = computeLikelihood(modelCR, rTst, MUi, COVi, SIGMAi, KAPPAi);

% COMPUTE POSTERIOR DISTRIBUTION
PPi = computePosteriorProbability(CLi, prior);

% READ OUT POSTERIOR
XHATi = readOutPosteriorCirc(Xi, PPi, method4est);

if bPLOT
    % ESTIMATE SUMMARY STATISTICS
    [XHATiMU, XHATiSD, XHATiCI68, XHATiCI90, XHATiCI95] = XhatByLevelCirc(XHATi, ...
      ctgIndTst, meanORmedian);

    % Fix angles that are ~360deg offset from the label
    estimateOffsets = XHATiMU(:) - X(:);
    if (any(abs(estimateOffsets)>180))
        ind2fix = find(abs(estimateOffsets)>180);
        offsetSign = -sign(estimateOffsets(ind2fix));
        XHATiMU(ind2fix) = XHATiMU(ind2fix) + offsetSign*360;
        XHATiCI68(:,ind2fix) = XHATiCI68(:,ind2fix) + offsetSign*360;
        XHATiCI90(:,ind2fix) = XHATiCI90(:,ind2fix) + offsetSign*360;
        XHATiCI95(:,ind2fix) = XHATiCI95(:,ind2fix) + offsetSign*360;
    end
        
    XHATiCI = XHATiCI68;

    figure('position',[560 485 1083 463]); 

    xPos = cosd(X-90);
    yPos = sind(X-90);
    uPos = cosd(XHATiMU-90)*0.5;
    vPos = squeeze(sind(XHATiMU-90)*0.5);
    quiver(xPos, yPos, uPos', vPos', 'k');
    vline(0, 'b');
    hline(0, 'b');
    l1 = refline(1,0);
    l1.Color='b';
    l2=refline(-1,0);
    l2.Color='b';

    subplot(1,3,1); hold on;
    plot(X,XHATiMU,'k','linewidth',1);
    errorbar(X,XHATiMU,XHATiCI(1,:)'-XHATiMU,XHATiCI(2,:)'-XHATiMU,'k-','linewidth',.5)
    axis(max(ceil(abs(X.*1.1)))*[-1 1 -1 1]);
    plot(xlim,ylim,'k--')
    formatFigure('X','Estimate of X');
    axis square

   % XHATiCIcirc = [XHATiCI, XHATiCI(:,1)]; 
   % Xcirc = [X, X(1)];
   % ciWidth = diff(XHATiCIcirc);
   % polarplot(Xcirc/360*2*pi, ciWidth);
   % pax = gca;
   % pax.ThetaZeroLocation = 'bottom';

   % subplot(1,3,2);
   % XHATiCIcirc = [XHATiCI, XHATiCI(:,1)]; 
   % Xcirc = [X, X(1)];
   % ciWidth = diff(XHATiCIcirc);
   % polarplot(Xcirc/360*2*pi, ciWidth);

    
    %subplot(1,3,2); hold on;
    %ciWidth = diff(XHATiCI);
    %ciWidth90 = diff(XHATiCI90);
    %polarplot(X, ciWidth, 'k-', 'linewidth', 1);
    %plot(X, ciWidth90, 'k--', 'linewidth', 1);
    %ylim(minmax(ceil(abs(diff(XHATiCI68)))).*[.2 5]);
    %set(gca,'yscale','log')
    %formatFigure('X','Confidence Interval');
    %axis square
    %
    %subplot(1,3,3); hold on;
    %bias = X(:) - XHATiMU(:); 
    %plot(X, bias,'ko-','linewidth',1);
    %formatFigure('X','Bias');
    %axis square
end
