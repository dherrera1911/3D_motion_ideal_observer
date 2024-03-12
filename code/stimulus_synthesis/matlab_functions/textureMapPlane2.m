function [Ipht, xSmp, ySmp] = textureMapPlane2(txtImg, txtZm, txtXm, txtYm, ppZm, ppXm, ...
  ppYm, xOffset, sDeg, tDeg, bPLOT)

% function textureMapPlane(txtImg,sDeg,tDeg,zp,txtXm,txtYm,IppZm,A,txtXm,txtYm,bPLOT)
%
%   example call: % MAP 1/F NOISE ONTO SLANTED SURFACE
%                   [X Y]=meshgrid(smpPos(256,1920),smpPos(256,1080)); 
%                   txtImg = pinkNoise(fliplr(size(X)),0);
%                   slntDeg = 30; tiltDeg = 0; 
%                   Ipht = textureMapPlane(txtImg,slntDeg,tiltDeg,5,X,Y,3,[],[],1);
%                   figure; imagesc([Ipht]); axis xy; axis image; colormap gray
%
%                 % MAP PLAID ONTO SLANTED SURFACE
%                   slntDeg = 70; tiltDeg = 90; IPDm = 0.0; 
%                   [txtXm txtYm IppZm] =  LRSIprojPlaneAnchorEye('C'); degPerMtr = 35.8./diff(minmax(txtXm));
%                   ZH = sin3(txtXm,txtYm,0.5.*degPerMtr,0,90,50,20,0); ZV = sin3(txtXm,txtYm,0.5.*degPerMtr,90,90,50,20,0); txtImg = mean(cat(3,ZH,ZV),3); 
%                   Lpht = textureMapPlane(txtImg,slntDeg,tiltDeg,3,txtXm,txtYm,3,-IPDm/2,txtXm,txtYm);
%                   Rpht = textureMapPlane(txtImg,slntDeg,tiltDeg,3,txtXm,txtYm,3,+IPDm/2,txtXm,txtYm);
%                   figure; imagesc([Lpht Rpht]); axis xy; axis image; colormap gray; 
%
%                 % MAP IMAGE ONTO SLANTED SURFACE
%                   textureMapPlane(Lpht,45,90,5,LppXm-IPDm/2,LppYm,3,0,LppXm-IPDm/2,LppYm,1);
%                   textureMapPlane(Lpht,45, 0,5,LppXm-IPDm/2,LppYm,3,0,LppXm-IPDm/2,LppYm,1);
%
% assumes that the center of the coordiante system is 
%                  
% txtImg:   image of texture in fronto-parallel plane         [ r x c ]
% sDeg:     slant in deg                                      [ 1 x 1 ]
% tDeg:     tilt  in deg                                      [ 1 x 1 ]
% zp:       distance of plane along optic axis                [ 1 x 1 ] 
% txtXm:    x-coordinates in meters in the projection plane   [ r x c ]
% txtYm:    y-coordinates in meters in the projection plane   [ r x c ]
% A:        x-coordinate  of eye at arbitrary position in x   [ 1 x 1 ]
%           LE -> A = -IPDm/2
%           RE -> A = +IPDm/2
% txtXm:    x-coordinates in meters of texture                [ r x c ]
% txtYm:    y-coordinates in meters of texture                [ r x c ]
% bPLOT:    1 -> plot
%           0 -> don't
% %%%%%%%%%%%%%%%%%%%%%
% Ipht:     image of texture mapped surface with specified geometry
% x0:       x-crds in fronto-parallel plane w. texture
% y0:       y-crds in fronto-parallel plane w. texture

if  min(size(ppXm))==1  && isempty(ppYm)      [ppXm, ppYm] = meshgrid(ppXm);       end
if  min(size(ppXm))==1  && min(size(ppYm))==1 [ppXm, ppYm] = meshgrid(ppXm,ppYm); end
if ~exist('xOffset','var')     || isempty(xOffset)       xOffset   = 0;     end
if ~exist('txtXm','var') || isempty(txtXm) txtXm = txtXm; end
if ~exist('txtYm','var') || isempty(txtYm) txtYm = txtYm; end
if ~exist('sDeg','var') || isempty(sDeg) sDeg = 0; end
if ~exist('tDeg','var') || isempty(tDeg) tDeg = 0; end
if ~exist('bPLOT','var') || isempty(bPLOT) bPLOT = 0;     end
if ~isempty(txtImg) && (size(txtImg,1) ~= size(txtXm,1) || size(txtImg,2) ~= size(txtXm,2))
  error(['textureMapPlane: WARNING! size(txtImg)]=[' num2str(size(txtImg)) ...
  '] does not equal size(txtXm)=[' num2str(size(txtXm)) ']']);
end

% Obtain the X-coordinates of the points that are sampled at the texture's Z-position
xSmp = (txtZm .* (ppXm-xOffset) .* cosd(tDeg) + txtZm .* ppYm .* sind(tDeg) + ...
           ppZm .* xOffset .* cosd(tDeg)) ./ ...
       (ppZm .* cosd(sDeg) - (ppXm - xOffset) .* cosd(tDeg) .* sind(sDeg) - ...
           ppYm .* sind(tDeg) .* sind(sDeg) );
% Obtain the Y-coordinates of the points that are sampled at the texture's Z-position
if     abs(cosd(tDeg)) >= 0.5
  ySmp = ( ppYm.*(xSmp.*sind(sDeg) + txtZm) - ppZm.*xSmp.*sind(tDeg).*cosd(sDeg) ) ...
    ./ (ppZm.*cosd(tDeg));
elseif abs(cosd(tDeg)) < 0.5
  ySmp = (ppZm.*(xSmp.*cosd(tDeg).*cosd(sDeg) - xOffset)  - ...
            (ppXm-xOffset).*(xSmp.*sind(sDeg) + txtZm) ) ...
         ./ (ppZm.*sind(tDeg));
end

% ROTATE TEXTURE COORDINATES ABOUT Z-AXIS (PRESERVES TEXTURE ASPECT RATIO)
if tDeg ~= 0
  R = [cosd(-tDeg) -sind(-tDeg); sind(-tDeg) cosd(-tDeg)]; % rotation matrix
  XX = [xSmp(:) ySmp(:)] * R; % rotate coordinates
  xSmp = reshape(XX(:,1), size(xSmp));
  ySmp = reshape(XX(:,2), size(xSmp));
end

% Verify that the sampled coordinates are within the texture's boundaries
xSmpMin = xSmp(1,1);
xSmpMax = xSmp(end,end);
ySmpMin = ySmp(1,1);
ySmpMax = ySmp(end,end);
txtXMin = min(txtXm(:));
txtXMax = max(txtXm(:));
txtYMin = min(txtYm(:));
txtYMax = max(txtYm(:));
if xSmpMin < txtXMin || xSmpMax > txtXMax || ySmpMin < txtYMin || ySmpMax > txtYMax
  error(['textureMapPlane: WARNING! sampled coordinates are outside of texture boundaries']);
end

%% Interpolate texture onto sampled coordinates
Ipht = interp2(txtXm, txtYm, txtImg, real(xSmp), real(ySmp));

if bPLOT
   figure; 
   imagesc(txtXm(1,:), txtYm(:,1)', Ipht); axis image; 
%    imagesc(Ipht.^.4); axis image; 
   formatFigure([],[],['\Deltax=' num2str(xOffset) '; \sigma=' num2str(sDeg) '; \tau=' num2str(tDeg)]);
   colormap gray; axis xy;
   if txtYm(1) > txtYm(end) axis xy; end
end

