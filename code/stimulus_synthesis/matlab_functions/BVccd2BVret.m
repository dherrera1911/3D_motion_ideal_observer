function [LretBV, RretBV, LccdBV, RccdBV] = BVccd2BVret(LccdBV, RccdBV, ...
  durationMs, upK, lensInfo, bPLOT)

% Get some relevant quantities
numSmp = size(LccdBV,4); % number of samples in movie
msPerSmp = durationMs./numSmp; % ms per sample
PszXY = size(LccdBV,[2,1]); % image size
% difference between image size and PSF size
PszRCdff = fliplr(PszXY - size(lensInfo.PSF_L, [2,1])); 
% Get index of temporal samples to keep
indSmp = upK:upK:numSmp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SENSOR TRANSFER FUNCTION % HI RES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TIR = coneTemporalImpulseResponse(msPerSmp.*[1:(numSmp)], 2, 0, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% OPTICS AND SENSOR TRANSFER FUNCTIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PSF_Lbffr = padarray(lensInfo.PSF_L, PszRCdff/2);
PSF_Rbffr = padarray(lensInfo.PSF_R, PszRCdff/2);
OTF_Lbffr = fftshift(fft2(fftshift(PSF_Lbffr)));
OTF_Rbffr = fftshift(fft2(fftshift(PSF_Rbffr)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY LENS OPTICS & TEMPORAL INTEGRATION FUNCTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE-TIME CONVOLUTION
LretBV = convtemporal(LccdBV, mean(LccdBV(:)), TIR, OTF_Lbffr, 1, 0);
RretBV = convtemporal(RccdBV, mean(RccdBV(:)), TIR, OTF_Rbffr, 1, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOWNSAMPLE TO DESIRED TEMPORAL RESOLUTION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LccdBV   = squeeze(LccdBV(:,:,1,indSmp-upK+1));
RccdBV   = squeeze(RccdBV(:,:,1,indSmp-upK+1));
LretBV   = squeeze(LretBV(:,:,1,indSmp));
RretBV   = squeeze(RretBV(:,:,1,indSmp));

%%%%%%%%%%%%%%%%
% PLOT RESULTS %
%%%%%%%%%%%%%%%%
if bPLOT
    %%
    figure(12121);
    set(gcf,'position',[600 150 500 600]);
    for i = 1:size(LretBV,length(size(LretBV)))
        imagesc([LretBV(:,:,i) RretBV(:,:,i); LccdBV(:,:,i) RccdBV(:,:,i)]);
        axis image; axis xy;
        colormap gray;
        caxis(minmax([LccdBV RccdBV]));
        formatFigure('RET',[],'CCD')
        set(gca,'xtick',[]); set(gca,'ytick',[]);
        % pause(5.*msPerSmp./1000);
        pause(.2);
    end
end

