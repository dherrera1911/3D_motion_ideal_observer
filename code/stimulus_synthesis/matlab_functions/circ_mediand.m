function [MUdeg] = circ_mediand(Adeg)

% function [MUdeg] = circ_median(Adeg)
%   
%   example call: circ_meand(Adeg)
%
% computes the median direction for circular data represented in degrees
%
% Adeg:     sample of angles in degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%
% MUdeg:   	median angle in degrees
%
% DHE 08/2022, modified from
% https://gitlab.unige.ch/Rodrigo.SalazarToro/matlab/-/blob/master/circ_stat/circ_median.m
% which is the circ_stat matlab package
%
% DHE comment: Existing version would handle some medians in an improper
% way, finding 2 tied directions, one pointing to the actual median, the
% other 180 degrees away from it, and average them to give a 'median'
% that was 90 degrees away from the actual median. We modify the code
% to better handle this case
%
%  Original code comment and attribution:
% References:
%   Biostatistical Analysis, J. H. Zar (26.6)
%
% Circular Statistics Toolbox for Matlab

% By Philipp Berens, 2009
% berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

if nargin < 2
  dim = 1;
end

% Added by daniel: Convert degrees to radians
alpha = Adeg*pi/180;

M = size(alpha);
med = NaN(M(3-dim),1);
for i=1:M(3-dim)
  if dim == 2
    beta = alpha(i,:)';
  elseif dim ==1
    beta = alpha(:,i);
  else
    error('circ_median only works along first two dimensions')
  end
  
  beta = mod(beta,2*pi);
  n = size(beta,1);

  dd = circ_dist2(beta, beta);
  m1 = sum(dd>=0,1);
  m2 = sum(dd<0,1);

  dm = abs(m1-m2);
  if mod(n,2)==1
    [m idx] = min(dm);
  else
    m = min(dm);
    idx = find(dm==m,2);
  end

  if m > 1
    warning('Ties detected.') %#ok<WNTAG>
  end

  %%%%%%%%%%%%% Daniel's modification to original code:
  if length(idx)==1
      md = circ_mean(beta(idx));
  else
      warning('Multiple medians, selecting closest to mean') %#ok<WNTAG>
      meanDirection = circ_mean(alpha);
      meanDirection = mod(meanDirection, 2*pi);
      [~, closestMedian] = min(abs(beta(idx)-meanDirection));
      md = beta(idx(closestMedian));
  end
  %%%%%%%%%%%%%

  if abs(circ_dist(circ_mean(beta),md)) > abs(circ_dist(circ_mean(beta),md+pi))
    md = mod(md+pi,2*pi);
  end
  
  med(i) = md;
end

MUdeg = md*180/pi;

if dim == 2
  med = med';
end

