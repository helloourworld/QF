clear all
clc

%**************************************************************************
% Choices
%**************************************************************************

trnwin  = 12;
gamma   = 2;
iSim =1000;

tic
P       = 12000; 
stdize  = 1;
lamlist = 10.^3;
saveon  = 1;
demean  = 0;
trainfrq= 1;

para_str = strcat('maxP-', num2str(P), '-trnwin-', num2str(trnwin), ...
    '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', ...
    num2str(demean), '-variableimportance-Yprdtmp');
X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt'};

pwd_str = pwd; % get the local paths
save_path = strcat('../Step1_Predictions/tryrff_v2_SeparateSims/', para_str);
mat_save_path = strcat(save_path, '/mat/');

aggregated_plot_save_path = strcat('./RFF_Empirical_figures/', para_str,'/');
mkdir(aggregated_plot_save_path);

%**************************************************************************
% Load simulation data
%**************************************************************************

filename = strcat([mat_save_path 'iSim1.mat']);
load(filename, 'T','Y', 'dates','lamlist');

% get nSim
files_listing   = dir([mat_save_path '/*.mat']);
nSim = size(files_listing, 1);

Yprdtmp_collect = nan(T,100, length(X_columns), nSim);
for s = 1:nSim
    disp(s);
    filename = strcat([mat_save_path '/iSim' num2str(iSim) '.mat']);
    load(filename, 'Yprdtmp');
    Yprdtmp_collect(:,:,:,s)   = Yprdtmp;
end

clearvars Yprdtmp
Yprdtmp = nanmean(Yprdtmp_collect, 4);

%**************************************************************************
% Draw plots in 5x3 shape
%**************************************************************************

X_columns_format = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'b/m', 'ntis', 'lag mkt'};
figure
for g=1:15
    xvec = linspace(-1,1,100)';
    yvec = nanmean(squeeze(Yprdtmp(:,:,g)));
    subplot(5,3,g)
   
    plot(xvec,yvec,'linewidth',2)
    if trnwin == 120
        set(gca,'ylim',[-0.005,0.025]);
    else
        set(gca,'ylim',[-0.005,0.025]);
    end
    set(gca,'fontname','TimesNewRoman','fontsize',16);
    hold on
    yline(0,'--','color',.5*ones(1,3));
    xline(0,'--','color',.5*ones(1,3));
    title(X_columns_format(g))
end
set(gcf,'Position',[1 1 800 1200]);
ytickformat('%.3f')

if saveon==1
    % save plot
    saveas(gcf,[aggregated_plot_save_path '/VariableImportance_1000Sims.eps'],'eps2c');
end
close 'all'

