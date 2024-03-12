function XHATi = readOutPosterior(Xi, PPi, method4est)

% function XHATi = readOutPosterior(Xi,PPi,method4est)
%
%   example call:  % READ OUT MMSE ESTIMATES (MEAN OF POSTERIOR)
%                    XHAT = readOutPosterior(X,PP,'MMSE');
%
%                  % READ OUT  MAP ESTIMATES (MAP  OF POSTERIOR)
%                    XHAT = readOutPosterior(X,PP,'MAP');
%
% readout posterior probability distribution via specified estimate
%
% Xi:         X values for which posterior is defined  [1     x nCtg]
% PPi:        posterior probability pf X               [nStim x nCtg]
% method4est: method for reading out the posterior probability
%             'MMSE' -> minium mean squared error
%             'MAP'  -> maximum a posteriori estimate
% %%%%%%%%%%%%%%%%%%%
% XHATi:      X estimate based on the posterior probability distribution

if strcmp(method4est, 'MMSE') 
    % EXPECTED VALUE OF X -> E( X | R )
    XHATi = circ_meand(Xi, PPi, 2);
elseif strcmp(method4est, 'MAP')
    % MAX A POSTERIORI    -> argmax_x[ P( X | R) ] 
    [~, indMax]=  max(PPi, [], 2);   % find max of posterior
    XHATi(:,1) = Xi(indMax);      % find corresponding X
%     disp(['readOutPosterior: WARNING! method4est=MAP may produce quantized estimates']);
%     disp(['                           given that you are using ' num2str(size(PPi,2)) ' distributions']);
%     disp(['                           increase number of interpolated distributions?']);
    
    % KLUGE TO UPSAMPLE POSTERIORS SO MAP CAN BE ESTIMATED WITHOUT MORE INTERPOLATED DISTRIBUTIONS 
    % indMax = [];  
    % upSampK = 64;
    % XinterpDense = linspace(min(Xi),max(Xi),upSampK*length(Xi)+1); % interpolate to avoid quantization issues
    % for i = 1:upSampK  
    %   ii = (i-1)*ceil(size(PPi,1)/upSampK) + [1:ceil(size(PPi,1)/upSampK)];
    %   if i == upSampK, ii(ii>size(PPi,1)) = []; end
    %   [~,indMaxBlock]=  max(interp1(Xi,PPi(ii,:)',XinterpDense,'spline'),[],1); 
    %   indMax = [indMax; indMaxBlock'];
    % end
    %  % FIND X AT MAX OF POSTERIOR
    % XHATi = XinterpDense(indMax);  
else
    error(['readOutPosterior: WARNING! unhandled method4est: ' method4est]);
end
