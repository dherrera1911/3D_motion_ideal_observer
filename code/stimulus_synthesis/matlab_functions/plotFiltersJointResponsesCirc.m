function plotFiltersJointResponsesCirc(bPltRsp, fPairs, X, sTrn, fTrn, ctgIndTrn, sTst, ...
  fTst, ctgIndTst, ctg2plt, rMax, fano, var0, axisLims, bPLOTellipse, maxPoints, colorCode, plotHist)

% function plotFiltersJointResponses(bPltRsp,fPairs,X,sTrn,fTrn,ctgIndTrn,sTst,fTst,ctgIndTst,ctg2plt,rMax,fano,var0,axisLims,bPLOTellipse)
%
%   example call: plotFiltersJointResponses(1,[1 2; 3 4],X,s,f,ctgInd);
%
%                 plotFiltersJointResponses(1,[1 2; 3 4],AMA.X,AMA.s,AMA.f,AMA.ctgInd,[],[],[],[1:3:19]);
%
% plot joint filter responses for specified filter pairs
%
% bPltRsp:      boolean indicating whether to scatter plot responses or not
%               1 -> scatter plot responses
%               0 -> don't (plots only rsp ellipses)
% fPairs:       filter pairs to plot                 [ nPairs  x    2    ]
% X:            category values                      [ 1       x  nCtg   ]
% sTrn:         stimuli           (training set)     [ d       x nStmTrn ]
% fTrn:         filters           (training set)     [ d       x    q    ]
% ctgIndTrn:    category indices  (training set)     [ nStmTrn x    1    ]
%               having nCtg unique values
% sTst:         stimuli           (  test   set)     [ d       x nStmTst ]
%               default = sTrn
% fTst:         filters           (  test   set)     [ d       x    q    ] 
%               default = fTrn
% ctgIndTst:    category indices  (  test   set)     [ nStmTst x    1    ]
%               default = ctgIndTrn
% rMax:         response max
% fano:         response fano factor
% var0:         response baseline variance
% axisLims:     manual control over axis limits (e.g. rMax*[-1 1 -1 1])
% bPLOTellipse: plot ellipse or not
%               1 -> plot 
%               0 -> not
% fracPlot      number between 0 and 1 indicating what proportion of points to plot

% INPUT HANDLING
if ~exist('sTst',        'var') || isempty(sTst)         sTst      = sTrn;              end
if ~exist('fTst',        'var') || isempty(fTst)         fTst      = fTrn;              end
if ~exist('ctgIndTst',   'var') || isempty(ctgIndTst)    ctgIndTst = ctgIndTrn;         end
if ~exist('ctg2plt'  ,   'var') || isempty(ctg2plt)      ctg2plt   = unique(ctgIndTst); end
if ~exist('rMax',        'var') || isempty(rMax)         rMax      = 1;                 end
if ~exist('fano',        'var') || isempty(fano)         fano      = 0;                 end
if ~exist('var0',        'var') || isempty(var0)         var0      = 0;                 end
if ~exist('X',           'var') || isempty(X)            X = unique(ctgIndTst);         end
if ~exist('bPLOTellipse','var') || isempty(bPLOTellipse) bPLOTellipse = 1;         end
if ~exist('maxPoints','var')    || isempty(maxPoints) maxPoints = 1000;         end
if ~exist('colorCode','var')    || isempty(colorCode) colorCode = 'circular';         end
if ~exist('plotHist','var')    || isempty(colorCode) plotHist = 1;         end

% COMPUTE FILTER RESPONSES TO TRAINING AND TEST STIMS

rspType = 'NUN';
if strcmp(rspType, 'NRW')
        %     FLT = FLTstruct([],'NPM','XYbino',[26 26]);
        %     % NRM = NRMstruct([],'AWB',[-0.01 3.82]);
        %     NRM = NRMstruct([],'BRD',[-0.5 1.91]);
        %     RSP = RSPstruct([],rMax,fano,var0,0);
        %     sAs = abs(fftVec(sTrn,'XYbino',[26,26,1])./size(sTrn,1));
        %     rTrn = stm2rsp(sTrn,sAs,fTrn,rMax,fano,var0,FLT,NRM,RSP,1);

        sRaw = bsxfun(@times, sTrn, sqrt(sum((sTrn).^2)) );
        sAS = abs(fftVec(sRaw, 'XYbino', [26,26,1])./size(sRaw,1));
        FLT = FLTstruct([], 'NPM', 'XYbino', [26 26]);
        NRM = NRMstruct([], 'AWB', [-0.01 3.82]);
        RSP = RSPstruct([], rMax, fano, var0, 0);
        rTrn = stm2rsp(sTrn, sAS, fTrn, rMax, fano, var0, FLT, NRM, RSP, 1);
elseif strcmp(rspType,'NUN')
    rTrn   = stim2resp(sTrn, fTrn, rMax);
end
rTst   = stim2resp(sTst, fTst, rMax);

% MLE FIT CONDITIONAL RESPONSE DISTIRBUTIONS
modelCR = 'gaussian';
[MU, COV, SIGMA, KAPPA] = fitCondRespDistribution(modelCR, [], [], rTrn, ctgIndTrn, rMax);

% PLOT FILTER RESPONSES
%figure('position', [493, 281,  0.5 .* size(fPairs,1).*810, 420]); 
figure('position', [100, 1200,  750, 750]); 

% SET COLORDER
% colors = getColorOrder([],2.*length(X));

if strcmp(colorCode, 'circular')
  colormapX = hsv(361); % the row is the color for a corresponding angle in degrees
  colorRowsX = round(X + abs(min(X)) + 1);
elseif strcmp(colorCode, 'back_forth')
  colormapX = parula(201);
  colorRowsX = round(cosd(X)*100) + 101;
elseif strcmp(colorCode, 'left_right')
  colormapX = parula(201);
  colorRowsX = round(sind(X)*100) + 101;
elseif strcmp(colorCode, 'linear_abs')
  colormapX = hsv(201);
  colorRowsX = round(abs(X)*100/max(X))+1;
elseif strcmp(colorCode, 'linear')
  colormapX = turbo(201);
  colorRowsX = round((X-min(X))*200/(max(X)-min(X)))+1;
end

for t = 1:size(fPairs,1)
    filtX = fPairs(t,1);
    filtY = fPairs(t,2);

    hold on; subplot(1, ceil(size(fPairs,1)), t); hold on
    % AXIS LIMS
    if ~exist('axisLims','var') || isempty(axisLims), if t == 1, axisLims = 1.2.*max(max(abs(rTrn(:,fPairs(t,:))))).*[-1 1 -1 1]; end; end
    % marginal histogram offset
    % PLOT CONDITIONAL DISTRIBUTIONS
    for c = 1:length(ctg2plt),
        % INDICES IN CURRENT CATEGORY
        ind = find(ctgIndTst==ctg2plt(c)); % & sqrt(sum(rTst(:,1:2).^2,2)) < 0.2);
        indRnd  = randsample(1:length(ind), min([maxPoints numel(ind)])); 
        % Unpack some variables for the category
        categoryColor = colormapX(colorRowsX(ctg2plt(c)),:);
        categoryXresps = rTst(ind(indRnd), filtX);
        categoryYresps = rTst(ind(indRnd), filtY);
        % PLOT ELLIPSE
        if bPLOTellipse
            CI = 90;
            h(c) = plotEllipse(MU(ctg2plt(c),:), COV(:,:,ctg2plt(c)), CI, fPairs(t,:), 2, categoryColor);
        end
        % SCATTER PLOT RESPONSES
        if bPltRsp
            % PLOT DATA POINTS
            if bPLOTellipse==0
                try
                h(c) = plot(categoryXresps, categoryYresps, 'wo', 'linewidth', .125, ...
                  'markerface', categoryColor, 'markersize', 6); 
                end
            else
                %plot(categoryXresps, categoryYresps, 'wo', 'linewidth', .125, ...
                %  'markerface', categoryColor, 'markersize', 6); 
                scatter(categoryXresps, categoryYresps, 'MarkerEdgeColor', categoryColor, ...
                  'LineWidth', .125, 'MarkerFaceColor', categoryColor, 'MarkerFaceAlpha', 0.35, ...
                  'MarkerEdgeAlpha', 0.35); 
                % PLOT MARGINAL CATEGORY-CONDITIONED MARGINAL
                if plotHist
                    [HX, BX] = hist(categoryXresps, linspace(axisLims(1), axisLims(2), 31));
                    [HY, BY] = hist(categoryYresps, linspace(axisLims(1), axisLims(2), 31));
                    plot(BX, 0.1 .* diff(axisLims(1:2)).*HX./max(HX) + axisLims(1), 'color', ...
                      categoryColor, 'linewidth', 1.5);
                    plot(0.1.*diff(axisLims(1:2)).*HY./max(HY) + axisLims(1), BY, 'color', ...
                      categoryColor, 'linewidth',1.5);
                end
            end
        end
        if c == length(ctg2plt)
       %     [HX,BX] = hist(rTst(:,filtX), linspace(axisLims(1), axisLims(2), 31));
       %     [HY,BY] = hist(rTst(:,filtY), linspace(axisLims(1), axisLims(2), 31));
       %     plot(BX, 0.2.*diff(axisLims(1:2)).*HX./max(HX) + axisLims(1), 'k');
       %     plot(0.2.*diff(axisLims(1:2)).*HY./max(HY) + axisLims(1), BY, 'k');
        end
        % MAKE PRETTY
        formatFigure(['F' num2str(fPairs(t,1)) ' response'], ['F' num2str(fPairs(t,2)) ' response'])
%          ['X=' num2str(X(ctg2plt(c)),'%.3f')])
        % SET AXIS LIMS
        %axis(axisLims); 
        axis square;
        %pause(0.7);
    end;
end

try   leg=legend(h, legendLabel('',round(X(ctg2plt),2),1,1), 'Location', 'NorthEastoutside', 'FontSize', 12);
catch leg=legend(legendLabel('',round(X(ctg2plt),2),1,1), 'Location', 'NorthEastoutside', 'FontSize', 12);
leg.ItemTokenSize=[5,1];
end
