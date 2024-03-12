function [h] = plotMultipleEllipses(fPairs, X, ctg2plt, COV, MU, CI, colorCode, alpha, showLegend)

if ~exist('showLegend', 'var') || isempty(showLegend) showLegend=true; end

if strcmp(colorCode, 'circular')
  colormapX = hsv(361); % the row is the color for a corresponding angle in degrees
  colorRowsX = round(X + abs(min(X)) + 1);
elseif strcmp(colorCode, 'back_forth')
  colormapX = cool(201);
  colorRowsX = round(cosd(X)*100) + 101;
elseif strcmp(colorCode, 'left_right')
  colormapX = cool(201);
  colorRowsX = round(sind(X)*100) + 101;
elseif strcmp(colorCode, 'linear')
  colormapX = hsv(201);
  colorRowsX = round(abs(X)*100/max(X))+1;
end


for c = 1:length(ctg2plt)
    categoryColor = colormapX(colorRowsX(ctg2plt(c)),:);
    categoryColor = [categoryColor, alpha];
    h(c) = plotEllipse(MU(ctg2plt(c),:), COV(:,:,ctg2plt(c)), ...
        CI, fPairs, 2, categoryColor);
    if showLegend
        leg=legend(h, legendLabel('',round(X(ctg2plt),2),1,1), 'Location', 'NorthEast', 'FontSize', 10);
    end
end

