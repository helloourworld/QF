
clear all
clc

%**************************************************************************
% Choices
%**************************************************************************
gamma       = 2;
stdize      = 1;
demean      = 0;
subsamp     = 1;
maxP        = 12000;
nSim = 1000;
saveon = 1;
trnwin_list      = [12, 60, 120];

pwd_str = pwd; % get the local paths
gybench_datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
combined_data_save_path = './combined_data/';
combined_data_high_complexity_save_path = './combined_data_high_complexity/';
mkdir('./RFF_Empirical_figures/');

% initialization
para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin_list(1)), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');
save_path = strcat(datadir, para_str);
filename = strcat([save_path '/iSim1.mat']);
load(filename, 'T','nP','nL','Y', 'Plist', 'log_lamlist','dates','lamlist');

VoC_pihat_collect = nan(T,length(trnwin_list));

%**************************************************************************
% Collect
%**************************************************************************

for trnwin_idx = 1:length(trnwin_list)
    trnwin = trnwin_list(trnwin_idx);
    
    para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');
    save_path = strcat(datadir, para_str);
    
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

    % collect pihat
    VoC_pihat_collect(:, trnwin_idx) = VoC_pihat;
    
end

%**************************************************************************
% Plot
%**************************************************************************
datesdt     = datetime(floor(dates/100),dates-floor(dates/100)*100,ones(size(dates)));

hold on
tmp1        = plot(datesdt,movavg(VoC_pihat_collect,'simple',6),'LineWidth',2);
hBands      = recessionplot;
set(hBands,'FaceColor',zeros(1,3),'FaceAlpha',0.2);
legend(['$\hat\pi, T = 12$'],['$\hat\pi, T = 60$'],['$\hat\pi, T = 120$'],...
'NBER Recession','interpreter','latex')
set(gcf,'Position',[1 1 1200 800]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.2, 0.6])
if saveon==1
    saveas(gcf,['./RFF_Empirical_figures/pihat-compare-across-T.eps'],'eps2c');
end
close 'all'


