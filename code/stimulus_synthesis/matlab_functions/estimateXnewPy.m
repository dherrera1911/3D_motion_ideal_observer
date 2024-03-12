function [XHAT, PP, CL] = estimateXnewPy(X, MU, COV, rTst, method4est, PR)

% function [XHAT,PP,CLi,Xi,MUi,COVi,SIGMAi,KAPPAi] = estimateXnew(modelCR,X,sTrn,fTrn,rTrn,ctgIndTrn,sTst,fTst,rTst,ctgIndTst,rMax,fano,var0,method4est,meanORmedian,method4interp,interpFactor,PR,bPLOT,CI)   
%                                                                              
%   example call: [s ctgInd     X] = loadAMAdevData('Disparity','Sml');
%                 [f E minTimeSec] = amaR01('GSS','MAP',2,0,1,[],s,ctgInd,X,1,.5,0.05,[]);
%                 [XHAT,PP,CLi,Xi,MUi,COVi,SIGMAi,KAPPAi] = estimateXnew('gaussian',X,s,f,[],ctgInd,[],[],[],[],rMax,fano,var0,'MAP','median','spline',8,[],1); )
%
% X:             latent variable values                  [  1   x nLvl ]
% MU:      mean of responses
% COV:       covariance of responses
% rTst:          responses         (  test  )            [ nDim x  nF  ]
% ctgIndTst:     category indices  (  test  )            [ nStm x  1   ]
% method4est:    method for reading out the posterior probability
%                 'MMSE' -> minium mean squared error
%                 'MAP'  -> maximum a posteriori estimate
% meanORmedian:   'mean'   plots the mean estimate
%                 'median' plots the median estimate
% PR:            prior                                   [   1 x numCtg ]
% bPLOT:         plot or not
%                1 -> plot
%                0 -> not
% CI:            confidence interval
%%%%%%%%%%%%%%%%%%%%%
% XHAT:      X estimate based on the posterior probability distribution
% Xi:                    X values of interpolated distributions
% CL:         likelihood                               [nStm x nCtg]
% PP:        posterior probability pf X               [nStm x nCtg]
% MU:         mean matrix        (    gaussian & gengauss model      )
%                                (               gengauss model      )
% COV:        covariance matrix  (    gaussian            model      )
%                                         
% SIGMA:      scale matrix       (               gengauss model      ) 
%                                ( ... see gengaussSigma2Cov.m   ... )
% KAPPA:      kappa matrix       (               gengauss model      ) 
%                                ( ... see gengaussKappa2Kurtosis.m  )

% COMPUTE/READ-IN RESPONSES
if ~exist('PR','var') || isempty(PR)
  PR = [];
end

modelCR = 'gaussian';

% COMPUTE LIKELIHOOD
CL = computeLikelihood(modelCR, rTst, MU, COV, [], []);

% COMPUTE POSTERIOR DISTRIBUTION
PP = computePosteriorProbability(CL, PR);

% READ OUT POSTERIOR
XHAT = readOutPosterior(X, PP, method4est);

