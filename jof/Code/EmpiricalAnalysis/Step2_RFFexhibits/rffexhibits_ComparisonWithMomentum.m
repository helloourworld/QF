
clear all
clc

trnwin      = 12;

%read data 
M = readtable('MS-20220538-misc(CRSPMonthly).xlsx');  
r = M{:,{'rvwind'}};
rf = M{:,{'rf'}}; 
ret = r-rf; 
Mdates = M{:,{'month'}};

vol_standardize_ret_first = 0; 
if vol_standardize_ret_first==1
    mstd = movstd(ret,[11 0],'Endpoints','discard');
    mstd = mstd(1:end-1,:);
    ret = ret(13:end,:);
    retstd = ret./mstd; % standardized returns
    mret = movmean(retstd,[trnwin-1 0],'Endpoints','discard');  % moving average of 12 periods
    mret_unstandardized = movmean(ret,[trnwin-1 0],'Endpoints','discard');
    Mdates = Mdates(13:end,:);
else
    mret_unstandardized = movmean(ret,[trnwin-1 0],'Endpoints','discard');  % moving average of 12 periods
    mstd = movstd(ret,[trnwin-1 0],'Endpoints','discard');   % moving std of 12 periods
    mret = mret_unstandardized./mstd; 
end

%form portfolio with weight based on standardized lagged mean returns 
w = mret(1:end-1); % lag 1 mom
p = w.*ret(trnwin+1:end); 
p_unstandardized = mret_unstandardized(1:end-1).*ret((trnwin+1):end); 

% use data from 1930 to match with OOS VoC strategy
Mdates = Mdates((trnwin+1):end); % we add 36 here because we used 36-month vol adjustment for signals
loc = find((Mdates >= datetime(1930,01,01)) & (Mdates <= datetime(2020,12,31)) );
Mdates = Mdates(loc);
p = p(loc);
p_unstandardized = p_unstandardized(loc);
mret = mret(1:end-1);
mret = mret(loc);
mret_unstandardized = mret_unstandardized(1:end-1);
mret_unstandardized = mret_unstandardized(loc);
ret = ret((trnwin+1):end);
ret = ret(loc);

%Sharpe ratio 
SR = sqrt(12)*mean(p)/std(p) % 0.3288

%Information ratio 
X = [ones(size(p)) ret]; 
beta = inv(X'*X)*X'*p; 
a = p - ret*beta(2); 
IR = sqrt(12)*mean(a)/std(a)  % 0.3426

%**************************************************************************
% Choices
%**************************************************************************
gamma       = 2;
stdize      = 1;
demean      = 0;
subsamp     = 1;
maxP        = 12000;
para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');
nSim = 1000;
saveon = 1;

pwd_str = pwd; % get the local paths
gybench_datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
figdir      = strcat('./RFF_Empirical_figures/', para_str,'/');
datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
save_path = strcat(datadir, para_str);
combined_data_save_path = './combined_data/';
combined_data_high_complexity_save_path = './combined_data_high_complexity/';

mkdir('./RFF_Empirical_figures/');
mkdir(figdir);
mkdir(combined_data_save_path);

if gamma == 0.5
    gamma_str = '0pt5';
else
    gamma_str = num2str(gamma);
end

%**************************************************************************
% Collect
%**************************************************************************
filename = strcat([save_path '/iSim1.mat']);
load(filename, 'T','nP','nL','Y', 'Plist', 'log_lamlist','dates','lamlist');

files_listing   = dir([save_path '/*.mat']);
nSim = size(files_listing, 1);

pihat_filename = strcat(combined_data_high_complexity_save_path, para_str, '.mat');
% get the VoC pihat of main analysis in the paper
if isfile(pihat_filename)
    load(pihat_filename); 
else    
    Yprd_collect    = nan(T,nP,nL,nSim); % predicted Y
    Bnrm_collect    = nan(T,nP,nL,nSim); % beta norm
    for s = 1:nSim
        disp(s);
        % load data
        filename = strcat([save_path '/' files_listing(s).name]);
        load(filename, 'Yprd', 'Bnrm');
        
        Yprd_collect(:,:,:,s)   = Yprd;
        Bnrm_collect(:,:,:,s)   = Bnrm;
    end
    clearvars Yprd Bnrm
    Yprd = Yprd_collect;
    Bnrm = Bnrm_collect;
    
    Bnrmbar     = nanmedian(Bnrm,4);
    Bnrmbar     = squeeze(nanmedian(Bnrmbar,1));
    timing      = Yprd.*Y';
    pihat       = [squeeze(nanmean(Yprd(:,nP,nL,:),4)) squeeze(nanmean(Yprd(:,1,1,:),4))];
    
    VoC_pihat = squeeze(nanmean(Yprd(:,nP,nL,:),4));
    VoC_pihat_allsims = squeeze(Yprd(:,nP,nL,:));
    
    disp('Data Loaded')
    save(pihat_filename,'VoC_pihat','VoC_pihat_allsims', '-v7.3');
end


pctlist     = [1 2.5 5 25 50 75 95 97.5 99];


%**************************************************************************
% Portfolio Evaluation: Full Sample
%**************************************************************************

if subsamp==1
    subbeg      = [1926 1926 1975];
    subend      = [2020 1974 2020];
else
    subbeg      = 1926;
    subend      = 2020;
end

nPct        = length(pctlist);
nSub        = length(subbeg);

ss = 1;
% Evaluation period
locev       = find(dates>=subbeg(ss)*100 & dates<=(subend(ss)+1)*100);

%% Regress our complex strategy on the return of 1 year momentum strategy on the market

% The mom strategy should be something like the following.  

% Just build a weight w_t, and look at the strategy return 
% r_{strat,t+1} = w_t r_{m,t+1}

% There are two types of w_t to try
% 1. w_t = 12-month average market return ending at t
% 2. w_t = 12-month average market return ending at t divided by some measure 
% of volatility (e.g. try something like 12 or 36 month volatility ending at t)

% Once this is done, just run regression r_{voc,t} = alpha + beta*r_{mom strat,t} + e 
% report alpha and alpha t-stat

% 1. w_t = 12-month average market return ending at t
w_t = mret;  

clip_at_zero = 0;
if clip_at_zero == 1
    w_t(w_t<0) = 0
end

% plot 
loc_dates     = find(~isnan(VoC_pihat));
VoC_pihat = VoC_pihat(loc_dates, :);
dates = dates(loc_dates, :);
Y_dates = Y(:,loc_dates)';
datesdt     = datetime(floor(dates/100),dates-floor(dates/100)*100,ones(size(dates)));

mom_timing = w_t .* ret; % p_unstandardized; % ret: 1032*1

[datesdt,idates,iMdates] = intersect(datesdt, Mdates);
VoC_pihat = VoC_pihat(idates, :);
w_t = w_t(iMdates, :);
ret = ret(iMdates, :);

pihat = [VoC_pihat w_t];
corr_with_pihat = corr(pihat)

% pihat_positions
VoC_pihat_allsims = VoC_pihat_allsims(loc_dates, :);
correlations = corr(VoC_pihat_allsims);  % gamma = 2, mean = 0.9729
correlations_avg = mean(mean(correlations)) % gamma = 1, 0.9953

ss          = 1;
suffix      = ['trnwin-' num2str(trnwin) '-gamma-' gamma_str '-stdize-' num2str(stdize) '-demean-' num2str(demean) '-' num2str(subbeg(ss)) '-' num2str(subend(ss))];

hold on
tmp1        = plot(datesdt,[pihat movavg(pihat,'simple',6)],'LineWidth',1);
tmp1(3).LineWidth = 2;
tmp1(4).LineWidth = 2;
tmp1(3).Color = tmp1(1).Color;
tmp1(4).Color = tmp1(2).Color;
tmp1(1).Color = [tmp1(1).Color 0.1];
tmp1(2).Color = [tmp1(2).Color 0.1];
hBands      = recessionplot;
set(hBands,'FaceColor',zeros(1,3),'FaceAlpha',0.2);
legend(['$\hat\pi$ ($c=' num2str(round(Plist(nP)/trnwin)) '$, $z=10^{' num2str(log10(lamlist(nL))) '}$)'],...
['$\hat\pi$ (standardized ' num2str(trnwin) '-month momentum)'],...
['$\hat\pi$ (6m MA, $c=' num2str(round(Plist(nP)/trnwin)) '$, $z=10^{' num2str(log10(lamlist(nL))) '}$)'],...
['$\hat\pi$ (6m MA, standardized ' num2str(trnwin) '-month momentum)'],...
'NBER Recession','interpreter','latex')
set(gcf,'Position',[1 1 1200 800]);
if trnwin==12
    set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.5, 1])
elseif trnwin==60
    set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.2, 0.6])
elseif trnwin==120
    set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.1, 0.5])
end
if saveon==1
    saveas(gcf,[figdir suffix '-pihat-compare-with-' num2str(trnwin) 'mom-' num2str(trnwin) 'StandardizedRANGEADJ.eps'],'eps2c');
end
close 'all'


%% Get stats
% timing = pihat.* CRSPvw returns
timing_VoC_Mom = pihat.*ret;

% normalize all strategies to have the same unconditional volatility of 20% per year
timing_VoC_Mom = timing_VoC_Mom./std(timing_VoC_Mom)*0.2/sqrt(12);
% check
timing_std = std(timing_VoC_Mom)*sqrt(12)  % 0.2000    0.2000
% mean
timing_ER = mean(timing_VoC_Mom)*12  % 0.0840    0.0639
% SR
timing_SR = sharpe(timing_VoC_Mom)*sqrt(12) % 0.4202    0.3195

% IR vs. Rm
X = [ones(size(timing_VoC_Mom,1),1) ret]; 
% y = VoC
beta = inv(X'*X)*X'*timing_VoC_Mom(:,1); 
a = timing_VoC_Mom(:,1) - ret*beta(2); 
IR_VoC = sqrt(12)*mean(a)/std(a);  % 0.3211
% y = 12-month momentum
beta = inv(X'*X)*X'*timing_VoC_Mom(:,2); 
a = timing_VoC_Mom(:,2) - ret*beta(2); 
IR_12Mom = sqrt(12)*mean(a)/std(a);  % 0.3296
IR = [IR_VoC IR_12Mom] % 0.3211    0.3296

%% run spanning regressions of each strategy on the other
corr_with_pihat(1,2) 

% R_{voc} on R_{mom}
stats   = regstats(timing_VoC_Mom(:,1), timing_VoC_Mom(:,2),'linear',{'tstat','rsquare','r'});
tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
    stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
disp(tstats)

% R_{mom} on R_{voc}
stats   = regstats(timing_VoC_Mom(:,2), timing_VoC_Mom(:,1),'linear',{'tstat','rsquare','r'});
tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
    stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
disp(tstats)

%%
disp('Load gybench');
load([gybench_datadir 'gybench-trnwin-' num2str(trnwin) '-stdize-' num2str(stdize)  '-demean-' num2str(demean) '.mat'])

% VOC + Linear Kitchen Sink
VOC    = VoC_pihat.*Y_dates;
LKS    = timing_gy(loc_dates,end);

loc     = find(~isnan(VOC + LKS));
timing_VoC_LKS = VOC(loc, :)./nanstd(VOC(loc, :)) + LKS(loc, :)./nanstd(LKS(loc, :));
timing_VoC_LKS = timing_VoC_LKS./std(timing_VoC_LKS)*0.2/sqrt(12);
SR_timing_VoC_LKS = sharpe(timing_VoC_LKS)*sqrt(12)

stats   = regstats(timing_VoC_LKS, Y_dates(loc, :),'linear',{'tstat','rsquare','r'});
tmp1res = stats.tstat.beta(1) + stats.r;
% out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];
IR_timing_VoC_LKS = sqrt(12)*sharpe(tmp1res)

% Double check
a = timing_VoC_LKS - Y(:,loc_dates)'*stats.tstat.beta(2); 
IR = sqrt(12)*mean(a)/std(a) 

%% Check bets
disp('Check bets')

% VoC_pihat = squeeze(nanmean(Yprd(:,nP,nL,:),4));
VoC_pihat = VoC_pihat(find(~isnan(VoC_pihat)),:);

% sum(VoC_pihat>0)/length(VoC_pihat)
sum(VoC_pihat<0)/length(VoC_pihat)

avg_neg_bet_magnitude = mean(VoC_pihat(VoC_pihat<0));
avg_pos_bet_magnitude = mean(VoC_pihat(VoC_pihat>0));
abs(avg_neg_bet_magnitude)/abs(avg_pos_bet_magnitude)

%% VoC strategies zero-out the expected return
disp('VoC strategies zero-out the expected return')
VOC_long_only = VoC_pihat;
VOC_long_only(VOC_long_only<0) = 0;
pihat_VOC_long = [VoC_pihat, VOC_long_only];
timing_VOC_long = pihat_VOC_long.*Y_dates;
timing_VOC_long = timing_VOC_long./std(timing_VOC_long)*0.2/sqrt(12);

stats   = regstats(timing_VOC_long(:,2), timing_VOC_long(:,1),'linear',{'tstat','rsquare','r'});
tmp1res = stats.tstat.beta(1) + stats.r;
IR = sqrt(12)*sharpe(tmp1res);
    
tstats = [sqrt(12)*sharpe(timing_VOC_long(:,2)) IR stats.tstat.beta(1)*12 stats.tstat.t(1) ...
    stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
disp(tstats)

mom12_long_only = w_t;
mom12_long_only(mom12_long_only<0) = 0;
pihat_VOClong_mom12long = [VOC_long_only, mom12_long_only];
timing_VOClong_mom12long = pihat_VOClong_mom12long.*Y_dates;
timing_VOClong_mom12long = timing_VOClong_mom12long./std(timing_VOClong_mom12long)*0.2/sqrt(12);

stats   = regstats(timing_VOClong_mom12long(:,1), timing_VOClong_mom12long(:,2),'linear',{'tstat','rsquare','r'});
tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
    stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
disp(tstats)

stats   = regstats(timing_VOClong_mom12long(:,2), timing_VOClong_mom12long(:,1),'linear',{'tstat','rsquare','r'});
tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
    stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
disp(tstats)


%% VOC strategies vs one-month mom

% specifically, report: 
% i) SR of 1m mom 
% ii) alpha (for 20% vol) and t(alpha) of 1m mom versus market 
% iii) alpha and t(alpha) of full voc vs 1m mom

if trnwin == 12
    w_t_1m      = Y(:, loc_dates-1)';
    pihat_VOC_1mMom = [VoC_pihat w_t_1m];
    corr_with_pihat = corr(pihat_VOC_1mMom)
    pihat_VOC_1mMom_timing = pihat_VOC_1mMom .* ret; % p_unstandardized; % ret: 1032*1
    
    % normalize all strategies to have the same unconditional volatility of 20% per year
    pihat_VOC_1mMom_timing = pihat_VOC_1mMom_timing./std(pihat_VOC_1mMom_timing)*0.2/sqrt(12);
    % check
    pihat_VOC_1mMom_timing_std = std(pihat_VOC_1mMom_timing)*sqrt(12);  % 0.2000    0.2000
    % mean
    pihat_VOC_1mMom_timing_ER = mean(pihat_VOC_1mMom_timing)*12;  % 0.0840    0.0639
    % SR: i) SR of 1m mom
    pihat_VOC_1mMom_timing_SR = sharpe(pihat_VOC_1mMom_timing)*sqrt(12) % 0.4202    0.2158
    
    % ii) alpha (for 20% vol) and t(alpha) of 1m mom versus market
    stats   = regstats(pihat_VOC_1mMom_timing(:,2), ret,'linear',{'tstat','rsquare','r'});
    tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
        stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
    disp(tstats)
    
    % iii) alpha and t(alpha) of full voc vs 1m mom
    stats   = regstats(pihat_VOC_1mMom_timing(:,1), pihat_VOC_1mMom_timing(:,2),'linear',{'tstat','rsquare','r'});
    tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
        stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
    disp(tstats)
    
    % iv) alpha and t(alpha) of 1m mom vs full voc
    stats   = regstats(pihat_VOC_1mMom_timing(:,2), pihat_VOC_1mMom_timing(:,1),'linear',{'tstat','rsquare','r'});
    tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) ...
        stats.tstat.beta(2) stats.tstat.t(2) stats.rsquare];
    disp(tstats)
    
end


%% VoC compare with GW variables: Table FI
% if we used each GW variable to build its own timing strategy, 
% then ran an alpha test of VOC on ALL 15 strategies at once 
% (this is a regression against 15 strategies at once), does VOC have alpha.  
% 
% This is of course an extremely conservative text because the betas will 
% be in-sample, thus overfit, and it will understate the true alpha.  
% but if alpha is STILL significant, it is very powerful
filename = strcat([save_path '/iSim1.mat']);
load(filename, 'X');
X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt'};
X_columns_all = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt', 'all', 'VoC ER'};

X_timing = X(:,loc_dates-1)'.*Y(:,loc_dates)';
VoC_timing = VoC_pihat.*Y(:,loc_dates)';

% all
% 1. In training sample 1:t, run regression 1=w'R+e, where R is the set of
% 15 timing strategies.  the resulting w are the in-sample tangency weights.  
% these need to be re-scaled so that the in-sample tan ptf has 20% ann vol.
% 2. Use these weights to built the realized OOS tan ptf return at t+1
% 3. Go back to step 1 and do it again with training sample 1:t+1
X_timing_all = nan(length(VoC_timing), 1);
trnwin_tanstart = 36;
for t=trnwin_tanstart+1:length(VoC_timing)
        
        trnloc  = 1:t-1;
        X_timing_trn    = X_timing(trnloc,:);
        ones_trn    = ones(t-1, 1);   
        X_timing_tst    = X_timing(t,:);        
        w = X_timing_trn\ones_trn;
        
        % re-scaled so that the in-sample tan ptf has 20% ann vol.
        ones_prd = X_timing_trn*w;
        vol_trn = std(ones_prd);
        w = w/vol_trn*0.2/sqrt(12);
        
        % Test
        X_timing_all(t,:)= X_timing_tst*w;
        
end

X_timing_all = X_timing_all./nanstd(X_timing_all)*0.2/sqrt(12);
VoC_timing = VoC_timing./std(VoC_timing)*0.2/sqrt(12);

stats   = regstats(VoC_timing(trnwin_tanstart+1:end, :), X_timing_all(trnwin_tanstart+1:end, :),'linear',{'tstat','rsquare','r'});
tmp1res = stats.tstat.beta(1) + stats.r;
IR = sqrt(12)*sharpe(tmp1res)
tstats_all = [stats.tstat.beta(1)*12 stats.tstat.t(1) stats.tstat.beta(2) stats.tstat.beta(2) IR stats.rsquare];

% Univariate regression
X_timing = X_timing./std(X_timing)*0.2/sqrt(12);
VoC_timing = VoC_timing./std(VoC_timing)*0.2/sqrt(12);

VoC_iGW_tstats = [];
for iGW = 1:15
    stats   = regstats(VoC_timing, X_timing(:, iGW),'linear',{'tstat','rsquare','r'});
    tmp1res = stats.tstat.beta(1) + stats.r;
    IR = sqrt(12)*sharpe(tmp1res);
    
    tstats = [stats.tstat.beta(1)*12 stats.tstat.t(1) stats.tstat.beta(2) stats.tstat.t(2) IR stats.rsquare];
    VoC_iGW_tstats = [VoC_iGW_tstats; tstats];
end

VoC_iGW_tstats = [VoC_iGW_tstats; tstats_all];
VoC_ER = [mean(VoC_timing)*12, nan(1, 5)];
VoC_iGW_tstats = [VoC_iGW_tstats; VoC_ER];

colNames = {'GW predictor','annualized alpha', 't-stats', 'beta', 'beta t-stats', 'IR', '$R^2$'};
VoC_iGW_tstats_table = array2table(VoC_iGW_tstats);
VoC_iGW_tstats_table = [table(X_columns_all'), VoC_iGW_tstats_table];
allVars = 1:width(VoC_iGW_tstats_table);
VoC_iGW_tstats_table = renamevars(VoC_iGW_tstats_table,allVars,colNames)

writetable(VoC_iGW_tstats_table, ['./RFF_Empirical_figures/ComparisonWithUnivariateTimingStrategies_trnwin' num2str(trnwin) '.csv'])


