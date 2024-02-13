function [] = GW_benchmark_function(trnwin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function runs the OLS benchmark with GW predictors
% trnwin: training window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%**************************************************************************
% Parameter Choices
%**************************************************************************

% Standardization = True
stdize  = 1;

% save the results
saveon  = 1;

% Demeaning = False
demean  = 0;

% shrinkage parameters lambda (z)
lamlist = [0 10.^([-3:1:3])];

% length of shrinkage parameters
nL      = length(lamlist);

%**************************************************************************
% Load Data
%**************************************************************************

load GYdata

% Add lag ret as a predictor
X       = [X lagmatrix(Y,[1])];


% Vol-standardize
if stdize==1
    % Standardize X
    X       = volstdbwd(X,[]);  % original volstdbwd0
    
    % Standardize Y
    Y2      = 0;
    for j=1:12
        Y2  = Y2+lagmatrix(Y.^2,[j]);
    end
    Y2      = Y2/12;
    Y       = Y./sqrt(Y2);
    clear Y2

    % Drop first 3 years due to vol scaling of X
    Y       = Y(37:end);
    X       = X(37:end,:);
    dates   = dates(37:end,:);
else
    loc     = find(~isnan(sum([X;Y])));
    X       = X(loc,:);
    Y       = Y(loc,:);
end

T       = length(Y);
X       = X';
Y       = Y';
d       = size(X,1);

%**************************************************************************
% Output Space
%**************************************************************************

Yprd    = nan(T,nL);
Bnrm    = nan(T,nL);

%**************************************************************************
% Recursive Estimation
%**************************************************************************

Yprd        = nan(T,nL);
Bnrm        = nan(T,nL);
for t=trnwin+1:T
    % time-rolling window data processing
    trnloc  = (t-trnwin):t-1;
    Ztrn    = X(:,trnloc);
    Ytrn    = Y(trnloc);
    Ztst    = X(:,t);
    if demean==1
        Ymn     = nanmean(Ytrn);
        Zmn     = nanmean(Ztrn,2);
    else
        Ymn     = 0;
        Zmn     = 0;
    end
    Ytrn    = Ytrn-Ymn;
    Ztrn    = Ztrn-Zmn;
    Ztst    = Ztst-Zmn;
    Zstd    = nanstd(Ztrn,[],2);
    Ztrn    = Ztrn./Zstd;
    Ztst    = Ztst./Zstd;

    % Train            
    B       = ridgesvd(Ytrn',Ztrn',lamlist*trnwin);

    % Test
    Yprd(t,:)= B'*Ztst + Ymn;
    Bnrm(t,:) = sum(B.^2);
end

%**************************************************************************
% Portfolio Evaluation Full Sample
%**************************************************************************

close 'all'
Bnrmbar     = nanmedian(Bnrm,4);
Bnrmbar     = squeeze(nanmedian(Bnrmbar,1));

% evaluation period
locev   = 1:length(Y);

timing  = Yprd.*Y';
SR      = squeeze(nanmean(squeeze(nanmean(timing,1)./nanstd(timing,[],1)),3));
nP=1;
ER      = nan(nL,1);
vol     = nan(nL,1);
IR      = nan(nL,1);
IRt     = nan(nL,1);
Ytmp    = Y(locev)';
for l=1:nL
    timtmp      = timing(locev,l);
    stats       = regstats(timtmp,Ytmp,'linear',{'tstat','r'});
    ER(l)    = nanmean(timtmp);
    vol(l)   = nanstd(timtmp);
    IR(l)    = stats.tstat.beta(1)/nanstd(stats.r);
    IRt(l)   = stats.tstat.t(1);
end
timing_gy   = timing;
Yprd_gy     = Yprd;
if saveon==1
    save_path = strcat('./tryrff_v2_SeparateSims/');
    save([save_path '/gybench-trnwin-' num2str(trnwin) '-stdize-' num2str(stdize)  '-demean-' num2str(demean) '.mat'],'timing_gy','Yprd_gy');
end

end

