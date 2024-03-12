function [XhatMU, XhatSD, XhatCI68, XhatCI90, XhatCI95] = XhatByLevelCirc(Xhat, ctgInd, meanORmedian, CI)

% function [XhatMU XhatSD XhatCI68 XhatCI90 XhatCI95] = XhatByLevel(Xhat,ctgInd,meanORmedian)
% 
%   example call: % TO COMPUTE MEAN X
%                   XhatByLevel(AMA.Xhat, AMA.ctgInd, AMA.X, 1)
%  
% mean estimate of X by level... returns mean, sd, 68%, 90%, and 95% confidence intervals
%
% Xhat:         vector of estimates [nx1]
% ctgInd:       vector of indices that indicates the level each estimate
%               belongs to          [nx1]
% meanORmedian: 'mean'   plots the mean estimate
%               'median' plots the median estimate
%%%%%%%%%%%%%%%%%%%%%%
% XhatMU:       mean (or median) of estimates
% XhatSD:       stddev of estimates
% XhatCI68:     68% confidence interval of estimates
% XhatCI90:     90% confidence interval of estimates
% XhatCI95:     95% confidence interval of estimates

if ~exist('meanORmedian','var') || isempty(meanORmedian)
   meanORmedian = 'mean'; 
end

warning off;
ctgIndUnq = unique(ctgInd);
for c = 1:length(ctgIndUnq)
    % FIND INDICES IN CURRENT CATEGORY
    % ind{c} = find(ctgInd == c);
    % ind{c} = find(ctgInd == c & isnan(Xhat) == 0);
    ind{c} = intersect(find(ctgInd == ctgIndUnq(c)), find(isnan(Xhat) == 0));

    % MEAN (or MEDIAN), STD, AND CI OF SIGNED ESTIMATES
    if strcmp(meanORmedian, 'mean')
        XhatMU(c,1) = circ_meand(Xhat(ind{c}));
    elseif strcmp(meanORmedian, 'median')
        XhatMU(c,1) = circ_mediand(Xhat(ind{c}));
    end

    XhatSD(c,1) = circ_stdd(Xhat(ind{c})); % stddev of angle

    % Compute distances between mean and individual samples
    % Because of how circular statistics work, and bimodality, we may
    % have the circular mean outside the confidence interval. If so, set the
    % bound of the CI to match the mean (that's what the min and max's do)
    Xdists = circ_distd(Xhat(ind{c}), XhatMU(c,1));
    XhatCI68(:,c) = XhatMU(c,1) + [min(quantile(Xdists, 0.16),0), max(quantile(Xdists, 0.84),0)];
    XhatCI90(:,c) = XhatMU(c,1) + [min(quantile(Xdists, 0.05),0), max(quantile(Xdists, 0.95),0)];
    XhatCI95(:,c) = XhatMU(c,1) + [min(quantile(Xdists, 0.025),0), max(quantile(Xdists, 0.975),0)];
end
warning on;
