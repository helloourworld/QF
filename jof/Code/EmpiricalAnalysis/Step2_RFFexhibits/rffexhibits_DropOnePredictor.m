clear all
clc
tic

%**************************************************************************
% Choices
%**************************************************************************

trnwin      = 12;
gamma       = 2;

stdize      = 1;
demean      = 0;
subsamp     = 0;
maxP        = 12000;
para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');
para_str_DropOneVariable = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-DropOneVariable');
nSim = 1000;
saveon = 1;
pctlist     = [1 2.5 5 25 50 75 95 97.5 99];
X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt'};

pwd_str = pwd; % get the local paths

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

%**************************************************************************
% Collect pihat for main analysis
%**************************************************************************
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

toc

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


%**************************************************************************
% Load simulation result for DropOneVariable
%**************************************************************************

pihat_filename = strcat(combined_data_high_complexity_save_path, ...
    para_str_DropOneVariable, '.mat');

% get the VoC pihat of main analysis in the paper
if isfile(pihat_filename)
    load(pihat_filename); 
else
    Yprd_collect_DropOneVariable    = nan(T,nSim, length(X_columns)); % predicted Y    
    for iGW = 1:length(X_columns)
        disp(iGW);
        for s = 1:nSim
            filename = char(strcat(datadir, para_str_DropOneVariable, '/', num2str(iGW),'_', X_columns(iGW), '/', files_listing(s).name));
            if isfile(filename)
                load(filename, 'Yprd');
                Yprd_collect_DropOneVariable(:,s,iGW)   = Yprd(:, nL);
            end
        end
    end
    clearvars Yprd
    Yprd_DropOneVariable = Yprd_collect_DropOneVariable;  %(T,nSim, length(X_columns))  
   
    disp('Data Loaded')
    save(pihat_filename, 'Yprd_DropOneVariable', '-v7.3');
end

%**************************************************************************
% Check performance of each predictor
%**************************************************************************
% Calculate the difference in the full model R2 versus each of the sub 
% model R2 (this is VIP_R2), and calculate the difference in the full 
% model SR versus each submodel R2 (this is VIP_SR).  
pihat_Paper_VoC = VoC_pihat;
pihat_DropOneVariable_VoC = squeeze(nanmean(Yprd_DropOneVariable, 2));

% R2
loc      = find(~isnan(Y'+pihat_Paper_VoC));
R2_Paper = 1-nanvar(Y(loc)'-pihat_Paper_VoC(loc))/nanvar(Y(loc)'); % 0.0059

R2_DropOneVariable = nan(length(X_columns), 1);
for iGW = 1:length(X_columns)
    R2_DropOneVariable(iGW, 1) = 1-nanvar(Y(loc)'-pihat_DropOneVariable_VoC(loc,iGW))/nanvar(Y(loc)'); % 0.0059
end
VIP_R2 = R2_Paper - R2_DropOneVariable;

% SR
SR_Paper = sharpe(pihat_Paper_VoC.*Y')*sqrt(12); 
SR_DropOneVariable = sharpe(pihat_DropOneVariable_VoC.*Y')*sqrt(12);
VIP_SR = SR_Paper - SR_DropOneVariable';

VIP_table = table(X_columns', VIP_R2, VIP_SR);
VIP_table = sortrows(VIP_table,3, 'descend');

if saveon==1
    VIP_table_filename = char(strcat(datadir, para_str_DropOneVariable));
    writetable(VIP_table,[VIP_table_filename '/VIP_table' num2str(trnwin) '.csv']);
end


%% Plot
X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
    'dp', 'dy', 'ltr', 'ep', 'b/m', 'ntis', 'lag mkt'};

[VIP_R2_Sorted,index] = sort(VIP_R2,'descend');
X_columns = categorical(X_columns(index));
X_columns = reordercats(X_columns,string(X_columns));

figure
yyaxis left
b = bar(X_columns,VIP_R2(index)*100);
b.FaceColor = [0, 0.4470, 0.7410];
ytickformat('percentage')

yyaxis right
p = plot(X_columns,VIP_SR(index), 'linewidth',3);
ytickformat('%.2f')

set(gcf,'Position',[1 1 1200 800]);
set(gca,'fontname','TimesNewRoman','fontsize',20)

yyaxis left
ylabel('VI ($R^2$)','interpreter','latex', 'fontsize',30)

yyaxis right
ylabel('VI (Sharpe Ratio)','interpreter','latex','fontsize',30)

saveas(gcf,[VIP_table_filename '/VIP_R2_SR_trnwin' num2str(trnwin) '.eps'],'eps2c');


