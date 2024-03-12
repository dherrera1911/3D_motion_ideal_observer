function [f] = plotInterpCov(Xi, COVi)
%%
disp(['interpCondRespDistribution: WARNING! rewrite plotting functions']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT COVARIANCE INTERPOLATION % (sanity check)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure('position', [861 15 1063 1069]);
[nr,nc] =subplotRowsCols(size(COVi,1).^2);
if nr*nc <= 64
    for r = 1:nr
        for c = r:nc
            i = (r-1)*nc + c;
            subplot(nr,nc,i);
            hold on;
            plot(Xi, squeeze(COVi(r,c,:)), 'k', 'linewidth', 2);
            %if r == c
            %    formatFigure('X','Variance',[],0,0,18,14)
            %else
            %    formatFigure('Deg','Covariance',[],0,0,18,14);
            %end
            yline(0);
            formatFigure('Deg','',[],0,0,18,14);
            set(gca,'ytick',[])
            set(gca,'xlabel',[])
        end
    end
end

