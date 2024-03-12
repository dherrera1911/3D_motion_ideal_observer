function [SDdeg] = circ_stdd(Adeg)

% function [SDdeg] = circ_median(Adeg)
%   
%   example call: SDdeg=circ_vmrnd(0,4,10); circ_stdd(Adeg)
%
% computes the circular standard deviation for circular data represented in degrees
%
% Adeg:     sample of angles in degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%
% SDdeg:   	standard deviation angle in degrees
%
% DHE 08/2022, modified from JDB: 08/17/2016
%
% References:
%   Statistical analysis of circular data, N. I. Fisher
%   Topics in circular statistics, S. R. Jammalamadaka et al. 
%   Biostatistical Analysis, J. H. Zar
%
% CONVERT TO RADIANS AND COMPUTE MEDIAN
    % COMPUTE MEDIAN
    [SDrad] = circ_std(Adeg.*pi./180);
    % CONVERT TO DEG
    SDdeg = SDrad.*180./pi;

