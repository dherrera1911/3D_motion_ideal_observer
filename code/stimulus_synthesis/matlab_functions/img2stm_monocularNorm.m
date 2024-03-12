function s = img2stm(I, W, sigEqv, bAddNse, nChnl, bNrmNB, Isz, bPreWdw)

% function s = img2stm(I,W,sigEqv,bAddNse,nChnl,bNrmNB,Isz,bPreWdw)
%
%   example call:
%
% convert intensity image to contrast normalized image
%
% I:          intensity image                               [ d x nStm ]
% W:          window                                        [ d x  1   ]
% sigEqv:     equivalent noise
% bAddNse:    1 -> do     add pixel noise to contrast image
%             0 -> do NOT add pixel noise to contrast image
%                  but normalize such that E[R] = r
% nChnl:      number of channels in which the normalization
%             should take place
% bNrmNB:     boolean indicating whether to use narrowband RMS contrast
% Isz:        size of the image before vectorization
% bPreWdw:    boolean indicating whether image has been pre-windowed
%             1 ->     pre-windowed
%             0 -> not pre-windowed     (default)
%%%%%%%%%%%%%%%%%%
% s:          contrast normalized stimulus

tol = 1e-1;

if sum(mean(I) < tol)>1 disp(['img2stm: WARNING! ' num2str(sum(mean(I)<tol)) ' of the I images are near mean = 0. I may be a contrast image, though it should be an intensity image']); end
if ~exist('W','var') | isempty(W) | W == 1,  W = ones(size(I,1),1);   end
if ~exist('sigEqv','var')  || isempty(sigEqv), sigEqv  = 0;           end 
if ~exist('bAddNse','var') || isempty(bAddNse) bAddNse = 0;           end
if ~exist('nChnl','var')   || isempty(nChnl)   nChnl   = 1;           end
if ~exist('bNrmNB','var')  || isempty(bNrmNB)  bNrmNB  = 0;           end
if ~exist('Isz','var')     || isempty(Isz)     Isz     = 0;           end
if ~exist('bPreWdw','var') || isempty(bPreWdw) bPreWdw = 0;           end

% CHANNEL INDICES TO NORMALIZE TOGETHER [ Nd/nChnl x nChnl ]
indNrm   = reshape(1:size(I,1),[],nChnl);
% ALLOCATE MEMORY
s = zeros(size(I));
for c = 1:nChnl
    % STIM DIMENSIONALITY IN CHANNEL
    Nd       = size(indNrm,1);
    % IMAGE MEAN
    if      bPreWdw == 0
        DC       = sum(bsxfun(@times,I(indNrm(:,c),:),W(indNrm(:,c))))./sum(W(indNrm(:,c)));
    elseif bPreWdw == 1
        DC       = mean( I(indNrm(:,c),:) );
    end
    % WEBER CONTRAST IMAGE: Iweb = (I - mean(I)) / mean(I)
    Iweb     = bsxfun(@rdivide,bsxfun(@minus,I(indNrm(:,c),:),DC),DC);
    % APPLY WINDOW TO CONTRAST IMAGE:  Iweb = Iweb.*W;
    Iweb     = bsxfun(@times,Iweb,W(indNrm(c))); 
    % COMPUTE PIXEL NOISE
    sigPix   = noiseEqv2noisePix( sigEqv, Nd );
    if bAddNse == 1
        %%%%%%%%%%%%%%%%%%%%%%
        % CONTRAST NORMALIZE %
        %%%%%%%%%%%%%%%%%%%%%%
        if bNrmNB == 0
            % BROADBAND NORMALIZATION (2D)
            s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( sum(Iweb.^2)) )];
        elseif bNrmNB == 1
            % NOISY INTENSITY IMAGE FROM NOISY CONTRAST IMAGE (REPROGRAM rmsContrastNB to do this!!!)
            Insy = bsxfun(@plus, bsxfun(@times,Iweb,DC), DC);
            % NARROWBAND / BROADBAND CONTRAST
            [IbNrmNB,IrmsBB] = rmsContrastNB([],Insy);
            % NARROWBAND NORMALIZATION (2D or 1D)
            s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( Nd.*(IbNrmNB.^2) ))];
            disp('img2stm: WARNING! untested performance');
        end
        % ADD PIXEL NOISE: Iweb = Iweb + N;
        N       = sigPix.*randn(size(Iweb));
        s(indNrm(:,c),:) = s(indNrm(:,c),:) + N;
    elseif bAddNse == 0
        % BASELINE NOISE
        c50 = sigPix;
        %%%%%%%%%%%%%%%%%%%%%%
        % CONTRAST NORMALIZE %
        %%%%%%%%%%%%%%%%%%%%%%
        if bNrmNB == 0
            % BROADBAND NORMALIZATION (IF 2D)
            s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( sum(Iweb.^2)  + Nd.*c50.^2 ))];
        elseif bNrmNB == 1
            % INTENSITY IMAGE FROM CONTRAST IMAGE  (REPROGRAM rmsContrastNB to do this!!!)
            Ichl = bsxfun(@plus, bsxfun(@times,Iweb,DC), DC);
            % NARROWBAND / BROADBAND CONTRAST
            [IbNrmNB,IrmsBB] = rmsContrastNB([],Ichl);
            % NARROWBAND NORMALIZATION
            s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( sqrt(Nd).*c50.^2 + Nd.*(IbNrmNB.^2) ))]; 
            % s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( sum(Iweb.^2)  + Nd.*c50.^2 + Nd.*(IbNrmNB.^2) ))];
            % s(indNrm(:,c),:) = [sqrt(1./nChnl).*bsxfun(@rdivide,Iweb,sqrt( 0.04.*sum(Iweb.^2) + 0.04.*Nd.*c50.^2 + 0.96.*Nd.*(IbNrmNB.^2) ))];
            disp('img2stm: WARNING! untested performance');
        end
    else
        error(['img2stm: WARNING! bAddNse has unhandled value: bAddNse = ' num2str(bAddNse)]);
    end
    
end


