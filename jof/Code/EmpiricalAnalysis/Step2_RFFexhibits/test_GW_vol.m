%**************************************************************************
% Check stationarity of raw predictors
% 1) normalize all GW predictors to have unit variance over the full sample, 
% 2) calculate the rolling 12-month volatility of each predictor, 
% 3) plot a bar chart of the time series average of this 12-month vol.
% from this we can pick the subset of signals with non-trivial variation 
% over 12-month samples, and redo our trading strategy.  this should be 
% good way to go about the analysis semyon suggests
% 4) produce a version for rolling 60-month and rolling 120-month vols
%**************************************************************************

clear all
clc

for trnwin = [12, 60, 120]
    % load predictors
    load ../Step1_Predictions/GYdata
    % trnwin = 120;
    
    % Add lag ret as a predictor
    X       = [X lagmatrix(Y,[1])];
    
    % 1) normalize all GW predictors to have unit variance over the full sample
    X = X(sum(isnan(X),2)==0,:);
    X = X./std(X);
    
    % 2) calculate the rolling 12-month volatility of each predictor
    X_mstd = movstd(X,[trnwin-1 0],'Endpoints','discard');
    
    % 3) plot a bar chart of the time series average of this 12-month vol
    X_mstd_mean = mean(X_mstd);
    X_columns = categorical({'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'b/m', 'ntis', 'lag mkt'});
    
    [X_mstd_mean_Sorted,index] = sort(X_mstd_mean,'descend');
    X_columns = reordercats(X_columns,string(X_columns(index)));
    bar(X_columns, X_mstd_mean)
    set(gcf,'Position',[1 1 1200 800]);
    set(gca,'fontname','TimesNewRoman','fontsize',20)
    saveon=1;
    if saveon==1
        figdir      = strcat('./RFF_Empirical_figures/');
        mkdir(figdir);
        saveas(gcf,[figdir 'GW1std_vol-' num2str(trnwin) '.eps'],'eps2c');
    end
    close 'all'
end
