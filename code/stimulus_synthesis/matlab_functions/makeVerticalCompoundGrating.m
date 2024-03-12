function [g, G] = makeVerticalCompoundGrating(x, y, frqCpd, A, phsDeg)

% implementation based on Geisler_GaborEquations.pdf in /VisionNotes
%
% x:          x position  in visual degrees matrix
%             []    -> 1 deg patch size, 128 smpPerDeg
%             [n]   -> 1 deg patch size,   n smpPerDeg
%             [1xn] -> x(end)-x(1) patch size, length(x) samples
% y:          y position  in visual degrees matrix       
% frqCpd:     frequency   in cycles per deg                    [1 x nComp ]
% A     :     amplitude 
%             [scalar] -> assigns same amplitude to all components
%             [1 x nComp] -> unique amplitude for each component
% ortDeg:     orientation in deg                               
%             [scalar] -> assigns same orientation to all components
%             [1 x nComp] -> unique orientation for each component
% phsDeg:     phase       in deg                                
%             [scalar] -> assigns same phase to all components
%             [1 x nComp] -> unique phase for each component
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% g:          gabor
% G:          gabor in frequency domain 
%             where G = fftshift(fft2(fftshift(g)))./sqrt(numel(g));

if isempty(x) || isempty(y)
   if isempty(x)
   [x, y] = meshgrid(smpPos(128,128));    
   elseif isscalar(x)
   [x, y] = meshgrid(smpPos(x,x));
   elseif isvector(x)
   [x, y] = meshgrid(x);    
   elseif min(size(x)) > 1
   y = x';    
   end
end

if isvector(x) && isvector(y) 
   [x, y]=meshgrid(x,y); 
end

% DEFAULT PARAMETER VALUES
if isempty(frqCpd); error('gabor2DcompoundBW: SPECIFY frqCpd!'); end
if isempty(A); A = ones([1 length(frqCpd)]); end
if isempty(phsDeg); phsDeg = zeros([1 length(frqCpd)]); end

% IF phsDeg IS SCALAR
if length(phsDeg)==1
   phsDeg = phsDeg.*ones([1 length(frqCpd)]); 
end

% IF A IS SCALAR
if length(A)==1
   A = A.*ones([1 length(frqCpd)]); 
end

% MAKE SURE PARAMETER VECTORS ARE SAME LENGTH
if ~(length(frqCpd)==length(phsDeg) && length(phsDeg)==length(A))
   error(['gabor2DcompoundBW: NUMBER OF ELEMENTS MUST BE SAME IN frqCpd, ortDeg,' ...
          ', phsDeg, and A!']); 
end

for i = 1:length(frqCpd)
    gCmp(:,:,i) = A(i).*cos( (2.*pi.*frqCpd(i).*x) + phsDeg(i).*pi./180);
end

% ADD COMPONENTS
g = sum(gCmp,3);

if nargout > 1 
    G = fftshift(fft2(ifftshift(g)))./sqrt(numel(g));
end

