function [] = rffexhibits_function(gamma, trnwin, stdize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The function generates exhibits in empirical analysis
% Parameters:
% gamma: gamma in Random Fourier Features
% trnwin: training window
% stdize: Standardization. stdize = 1 means True
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

%**************************************************************************
% Choices of parameters
%**************************************************************************
if gamma == 0.5
    gamma_str = '0pt5';
else
    gamma_str = num2str(gamma);
end

% Demeaning = False
demean      = 0;

% Use subsample
subsamp     = 1;

% max number of Random Fourier Features (RFFs)
maxP        = 12000;

% saving string
para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');

% the number of simulations is 1000
nSim = 1000; 

% save the results
saveon = 1;

% Local path
gybench_datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
figdir      = strcat('./RFF_Empirical_figures/', para_str,'/');
datadir     = '../Step1_Predictions/tryrff_v2_SeparateSims/';
save_path = strcat(datadir, para_str);
combined_data_save_path = './combined_data/';

% build the output folders
mkdir(figdir);
mkdir(combined_data_save_path);

%**************************************************************************
% Load parameters and benchmark
%**************************************************************************
filename = strcat([save_path '/iSim1.mat']);
load(filename, 'T','nP','nL','Y', 'Plist', 'log_lamlist','dates','lamlist');

files_listing   = dir([save_path '/*.mat']);

nSim = size(files_listing, 1);
Yprd_collect    = nan(T,nP,nL,nSim); % predicted Y
Bnrm_collect    = nan(T,nP,nL,nSim); % beta norm

%**************************************************************************
% Collect results of 1000 simulations
%**************************************************************************

% load the benchmark of Welch and Goyal (2008) "kitchen sink" regression
load([gybench_datadir 'gybench-trnwin-' num2str(trnwin) '-stdize-' num2str(stdize)  '-demean-' num2str(demean) '.mat'])

Y_B_file_save = ['trnwin-' num2str(trnwin) '-gamma-' num2str(gamma) '-stdize-' num2str(stdize) '-demean-' num2str(demean) '-Y-B'];
filename = strcat(combined_data_save_path, Y_B_file_save, '.mat');
if isfile(filename)
    % if the combined result exists
    load(filename);
else
    % if the combined result doesn't exist
    for s = 1:nSim
        disp(s); % random seed
        % load data
        filename = strcat([save_path '/' files_listing(s).name]);
        load(filename, 'Yprd', 'Bnrm');
        
        Yprd_collect(:,:,:,s)   = Yprd;
        Bnrm_collect(:,:,:,s)   = Bnrm;
    end
    
    clearvars Yprd Bnrm
    Yprd = Yprd_collect;
    Bnrm = Bnrm_collect;
    
    Bnrmbar     = nanmean(Bnrm,4);  % nanmedian(Bnrm,4);
    Bnrmbar     = squeeze(nanmean(Bnrmbar,1)); % squeeze(nanmedian(Bnrmbar,1));
    timing      = Yprd.*Y';
    pihat       = [squeeze(nanmean(Yprd(:,nP,nL,:),4)) squeeze(nanmean(Yprd(:,1,1,:),4))];
end

disp('Data Loaded')
toc

% percentile list
pctlist     = [1 2.5 5 25 50 75 95 97.5 99];

%% Generate exhibits
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

for ss=1:nSub
    % Evaluation period
    locev       = find(dates>=subbeg(ss)*100 & dates<=(subend(ss)+1)*100);
    
    % Suffix
    suffix      = ['trnwin-' num2str(trnwin) '-gamma-' num2str(gamma) '-stdize-' num2str(stdize) '-demean-' num2str(demean) '-' num2str(subbeg(ss)) '-' num2str(subend(ss))];
    performance_filename = strcat(combined_data_save_path, suffix, '.mat');
    
    % generate performance measurements
    if isfile(performance_filename)
        load(performance_filename);
    else

        %**************************************************************************
        % Performance initialization
        %**************************************************************************
        ER          = nan(nP,nL);
        SR          = nan(nP,nL);
        vol         = nan(nP,nL);
        IR          = nan(nP,nL);
        IRt         = nan(nP,nL);
        alpha       = nan(nP,nL);
        R2          = nan(nP,nL);
        IRstd       = nan(nP,nL);

        ERpct       = nan(nP,nL,nPct);
        SRpct       = nan(nP,nL,nPct);
        volpct      = nan(nP,nL,nPct);
        IRpct       = nan(nP,nL,nPct);
        IRtpct      = nan(nP,nL,nPct);
        alphapct    = nan(nP,nL,nPct);
        R2pct       = nan(nP,nL,nPct);
        Ytmp        = Y(locev)';
        for p=1:nP
            for l=1:nL
                ERtmp           = nan(nSim,1);
                voltmp          = nan(nSim,1);
                IRtmp           = nan(nSim,1);
                IRttmp          = nan(nSim,1);
                alphatmp        = nan(nSim,1);
                R2tmp           = nan(nSim,1);
                IRstdtmp        = nan(nSim,1);
                timtmp          = squeeze(timing(locev,p,l,:));
                Yprdtmp         = squeeze(Yprd(locev,p,l,:));
                
                parfor i=1:nSim
                    stats       = regstats(timtmp(:,i),Ytmp,'linear',{'tstat','r'});
                    SRtmp(i)    = sharpe(timtmp(:,i));
                    ERtmp(i)    = nanmean(timtmp(:,i));
                    voltmp(i)   = nanstd(timtmp(:,i));
                    IRtmp(i)    = stats.tstat.beta(1)/nanstd(stats.r);
                    IRttmp(i)   = stats.tstat.t(1);
                    alphatmp(i) = stats.tstat.beta(1);
                    IRstdtmp(i) = nanstd(stats.r);
                    
                    loc     = find(~isnan(Yprdtmp(:,i)+Ytmp));
                    R2tmp(i)    = 1-var(Yprdtmp(loc,i)-Ytmp(loc),'omitnan')/var(Ytmp(loc),'omitnan');            
                end
                SR(p,l)         = nanmean(SRtmp);
                ER(p,l)         = nanmean(ERtmp);
                vol(p,l)        = nanmean(voltmp);
                IR(p,l)         = nanmean(IRtmp);
                IRt(p,l)        = nanmean(IRttmp);
                alpha(p,l)      = nanmean(alphatmp);
                R2(p,l)         = nanmean(R2tmp);
                IRstd(p,l)      = nanmean(IRstdtmp);
                
                SRpct(p,l,:)    = prctile(SRtmp,pctlist);
                ERpct(p,l,:)    = prctile(ERtmp,pctlist);
                volpct(p,l,:)   = prctile(voltmp,pctlist);
                IRpct(p,l,:)    = prctile(IRtmp,pctlist);
                IRtpct(p,l,:)   = prctile(IRttmp,pctlist);
                alphapct(p,l,:) = prctile(alphatmp,pctlist);
                R2pct(p,l,:)    = prctile(R2tmp,pctlist);    
                disp(['p=' num2str(p) ', l=' num2str(l)])
            end
        end

        % save the data
        save(performance_filename,'locev','ER','SR', 'vol','IR','IRt','alpha','R2',...
            'ERpct','SRpct', 'volpct','IRpct','IRtpct','alphapct','R2pct', 'IRstd',...
            'Bnrmbar','timing', 'pihat', 'Yprd', '-v7.3')
    end

    % R2
    plot(Plist/trnwin,R2,'linewidth',1.5);line([1,1],[-10,1],'LineStyle','--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex');
    lh = legend([cellstr(strcat(strcat('$\log_{10}(z)=',num2str(log10(lamlist)')),'$'));{'$c=1$'}],'interpreter','latex'); 
    lh.Position(1) = 0.72 - lh.Position(3)/2; 
    lh.Position(2) = 0.4 - lh.Position(4)/2;
    set(gcf,'Position',[321 241 512 384]);
    if gamma == 2
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-10,.1])
    elseif gamma == 1 && trnwin == 12
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-10,.2])
    elseif gamma == 0.5 && trnwin == 60 && ss <3
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-10,.25])
    else
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-10,max(max(R2)) + 0.1])
    end

    ytickformat('%.1f')
    if saveon==1
        saveas(gcf,[figdir suffix '-R2.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis_legend([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis_legend([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis_legend([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-R2-brkaxis.eps'],'eps2c');
    end
    
    close 'all'

    % R2 zoom-in
    plot(Plist/trnwin,R2,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
    legend([cellstr(strcat(strcat('$\log_{10}(z)=',num2str(log10(lamlist)')),'$'));{'$c=1$'}],'interpreter','latex','location','southeast')
    set(gcf,'Position',[321 241 512 384]);
    if gamma == 2 && trnwin == 12
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.1,.1])
    elseif gamma == 1 && trnwin == 12
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.1,.2])
    else
        set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.1,max(max(R2)) + 0.01])
    end
    ytickformat('%.2f')
    if saveon==1
        saveas(gcf,[figdir suffix '-R2zoom.eps'],'eps2c');
    end
    close 'all'
    
    % Sharpe ratio
    plot(Plist/trnwin,sqrt(12)*SR,'linewidth',1.5);
    line([1,1],[min(min(sqrt(12)*SR)) - 0.01,max(max(sqrt(12)*SR)) + 0.01],'LineStyle','--','color',.5*ones(1,3));
    xlabel('$c$','interpreter','latex')
    
    set(gcf,'Position',[321 241 512 384]);
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(sqrt(12)*SR)) - 0.01,max(max(sqrt(12)*SR)) + 0.01])
    % end
    ytickformat('%.2f')
    if saveon==1
        saveas(gcf,[figdir suffix '-SR.eps'],'eps2c');
        lh = legend([cellstr(strcat(strcat('$\log_{10}(z)=',num2str(log10(lamlist)')),'$'));{'$c=1$'}],'interpreter','latex'); 
        lh.Position(1) = 0.72 - lh.Position(3)/2; 
        lh.Position(2) = 0.4 - lh.Position(4)/2;
    
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis_legend([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis_legend([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis_legend([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-SR-brkaxis.eps'],'eps2c');
    end
    
    close 'all'

    
    % ER
    plot(Plist/trnwin,ER,'linewidth',1.5); xlabel('$P/T$','interpreter','latex');
    set(gcf,'Position',[321 241 512 384]);
    if gamma == 2 && trnwin == 12 && ss == 1
        ylim_range = [0,0.035];
    elseif gamma == 1 && trnwin == 12 && ss == 1
        ylim_range = [0,0.06];
    elseif gamma == 0.5 && trnwin == 12 && ss == 3
        ylim_range = [min(min(ER))- 0.001,max(max(ER))+ 0.005];
    elseif gamma == 0.5 && trnwin == 12 && ss<3
        ylim_range = [0,max(max(ER))+ 0.005];
    else
        ylim_range = [max(0,min(min(ER))- 0.001),max(max(ER))+ 0.003];
    end
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',ylim_range);
    ytickformat('%.2f')
    
    if subbeg(ss)==1975 && subend(ss)==2020 && gamma > 0.5 && trnwin == 12
        set(gca,'ytick',[0,.01,.02,.03])
    elseif subbeg(ss)==1975 && subend(ss)==2020 && gamma == 0.5 && trnwin == 12
        set(gca,'ytick',[-0.04, -0.02, 0, .02, .04])
    end
    
    if subbeg(ss)==1926 % && subend(ss)==1974
        % When there is legend, break separately to avoid issue in breakxaxis 
        line([1,1],ylim_range,'LineStyle','--','color',.5*ones(1,3));
        lh = legend([cellstr(strcat(strcat('$\log_{10}(z)=',num2str(log10(lamlist)')),'$'));{'$P/T=1$'}],'interpreter','latex'); 
        lh.Position(1) = 0.72 - lh.Position(3)/2; 
        lh.Position(2) = 0.4 - lh.Position(4)/2;
        
        if saveon==1
            saveas(gcf,[figdir suffix '-ER.eps'],'eps2c');
            if max(Plist/trnwin) == 100
                set(gca,'xtick',[0,5,10,95,100])
                breakxaxis_legend([11 94], 0.05);
            elseif max(Plist/trnwin) == 1000
                set(gca,'xtick',[0:10:50,990,1000])
                breakxaxis_legend([51 989], 0.05);
            elseif max(Plist/trnwin) == 200
                set(gca,'xtick',[0,5,10,195,200])
                breakxaxis_legend([11 194], 0.05);
            end
            saveas(gcf,[figdir suffix '-ER-brkaxis.eps'],'eps2c');
        end
        
    else
        xline(1,'--','color',.5*ones(1,3));        
        if saveon==1
            saveas(gcf,[figdir suffix '-ER.eps'],'eps2c');
            if max(Plist/trnwin) == 100
                set(gca,'xtick',[0,5,10,95,100])
                breakxaxis([11 94], 0.05);
            elseif max(Plist/trnwin) == 1000
                set(gca,'xtick',[0:10:50,990,1000])
                breakxaxis([51 989], 0.05);
            elseif max(Plist/trnwin) == 200
                set(gca,'xtick',[0,5,10,195,200])
                breakxaxis([11 194], 0.05);
            end
            saveas(gcf,[figdir suffix '-ER-brkaxis.eps'],'eps2c');
        end 
    end
    close 'all'
    

    % Vol
    plot(Plist/trnwin,vol,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
    set(gcf,'Position',[321 241 512 384]);
    set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,min(5, max(max(vol))+ 0.1)])
    ytickformat('%.2f')
    if saveon==1
        saveas(gcf,[figdir suffix '-vol.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-vol-brkaxis.eps'],'eps2c');        
    end
    close 'all'

    % Info ratio
    plot(Plist/trnwin,sqrt(12)*IR,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
    set(gcf,'Position',[321 241 512 384]);
    if gamma == 2 && trnwin == 12 && ss ==1
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,0.33])
    elseif gamma == 1 && trnwin == 12 && ss == 1
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,0.3])
    elseif gamma == 0.5 && trnwin == 12 && ss<3
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,max(max(sqrt(12)*IR)) + 0.01])
    elseif gamma == 0.5 && trnwin == 12 && ss == 3
    set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(sqrt(12)*IR))- 0.01,max(max(sqrt(12)*IR)) + 0.01])
    else
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(sqrt(12)*IR))- 0.01,max(max(sqrt(12)*IR)) + 0.01])
    end
    ytickformat('%.2f')
    if saveon==1
        
        saveas(gcf,[figdir suffix '-IR.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-IR-brkaxis.eps'],'eps2c');        
    end
    close 'all'

    % Alpha
    plot(Plist/trnwin,alpha,'linewidth',1.5); xlabel('$P/T$','interpreter','latex')
    set(gcf,'Position',[321 241 512 384]);
    
    if gamma == 0.5 && trnwin == 12 && ss == 3
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(alpha))- 0.01,max(max(alpha)) + 0.01])
    elseif gamma == 0.5 && trnwin == 12 && ss == 1
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0, 0.03])
        set(gca,'ytick',[0,.01,.02,.03])
    elseif max(max(alpha)) > 0.015 && max(max(alpha)) <=0.025
        set(gca,'ytick',[0, 0.01, 0.02])
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,max(max(alpha)) + 0.002])
    elseif min(min(alpha))> 0 && max(max(alpha)) <=0.015
        set(gca,'ytick',[0, 0.01])
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,max(max(alpha)) + 0.001])
    else
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(alpha))- 0.001,max(max(alpha)) + 0.002])
    end
    ytickformat('%.2f')
    if saveon==1
        
        % When there is legend, break separately to avoid issue in breakxaxis 
        line([1,1],ylim_range,'LineStyle','--','color',.5*ones(1,3));
        lh = legend([cellstr(strcat(strcat('$\log_{10}(z)=',num2str(log10(lamlist)')),'$'));{'$P/T=1$'}],'interpreter','latex'); 
        lh.Position(1) = 0.72 - lh.Position(3)/2; 
        lh.Position(2) = 0.4 - lh.Position(4)/2;
        
        saveas(gcf,[figdir suffix '-alpha.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis_legend([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis_legend([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis_legend([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-alpha-brkaxis.eps'],'eps2c');        
    end
    close 'all'

    % Alpha t-stat
    plot(Plist/trnwin,IRt,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$P/T$','interpreter','latex')
    set(gcf,'Position',[321 241 512 384]);
    if gamma == 0.5 && trnwin == 12 && ss < 3
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[0,max(max(IRt)) + 0.1]);
    else
        set(gca,'fontname','TimesNewRoman','fontsize',20, 'ylim',[min(min(IRt)) - 0.1,max(max(IRt)) + 0.1]);
    end
    ytickformat('%.2f')
    if saveon==1
        saveas(gcf,[figdir suffix '-IRt.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-IRt-brkaxis.eps'],'eps2c');        
    end
    close 'all'

    % Bnorm
    plot(Plist/trnwin,Bnrmbar,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
    set(gcf,'Position',[321 241 512 384]);
    set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,3])
    ytickformat('%.2f')
    if saveon==1
        saveas(gcf,[figdir suffix '-Bnorm.eps'],'eps2c');
        if max(Plist/trnwin) == 100
            set(gca,'xtick',[0,5,10,95,100])
            breakxaxis([11 94], 0.05);
        elseif max(Plist/trnwin) == 1000
            set(gca,'xtick',[0:10:50,990,1000])
            breakxaxis([51 989], 0.05);
        elseif max(Plist/trnwin) == 200
            set(gca,'xtick',[0,5,10,195,200])
            breakxaxis([11 194], 0.05);
        end

        saveas(gcf,[figdir suffix '-Bnorm-brkaxis.eps'],'eps2c');
    end
    close 'all'
    
end

%%
%**************************************************************************
% Timing weights
%**************************************************************************

ss          = 1;
suffix      = ['trnwin-' num2str(trnwin) '-gamma-' gamma_str '-stdize-' num2str(stdize) '-demean-' num2str(demean) '-' num2str(subbeg(ss)) '-' num2str(subend(ss))];
datesdt     = datetime(floor(dates/100),dates-floor(dates/100)*100,ones(size(dates)));
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
legend(['$\hat\pi$, ($c=' num2str(round(Plist(nP)/trnwin)) '$, $z=10^{' num2str(log10(lamlist(nL))) '}$)'],...
['$\hat\pi$, ($c=' num2str(round(Plist(1)/trnwin,1)) '$, $z=10^{' num2str(log10(lamlist(1))) '}$)'],...
['$\hat\pi$ (6m MA, $c=' num2str(round(Plist(nP)/trnwin)) '$, $z=10^{' num2str(log10(lamlist(nL))) '}$)'],...
['$\hat\pi$ (6m MA, $c=' num2str(round(Plist(1)/trnwin,1)) '$, $z=10^{' num2str(log10(lamlist(1))) '}$)'],...
'NBER Recession','interpreter','latex')
set(gcf,'Position',[1 1 1200 800]);
if gamma == 2 && trnwin == 12
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.3,.7])
elseif gamma == 1 && trnwin == 12
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.6,1.4])
elseif gamma == 0.5 && trnwin == 60
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.5,1])
elseif gamma == 0.5 && trnwin == 12
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-1,1.5])
elseif gamma == 2 && trnwin == 60
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.1,0.3])
elseif gamma == 1 && trnwin == 60
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.2,0.6])
elseif gamma == 1 && trnwin == 120
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.1,0.4])
elseif gamma == 2 && trnwin == 120
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.05,0.2])
elseif gamma == 3 && trnwin == 12
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-0.1,0.4])
end
if saveon==1
    saveas(gcf,[figdir suffix '-pihat.eps'],'eps2c');
end
close 'all'

%**************************************************************************
% Generate Table 1: Comparison with Welch and Goyal (2008) and mkt
%**************************************************************************

out     = [];

% Mkt
stats   = regstats(Y',ones(size(Y')),1,'tstat');
out     = [out ; nan sqrt(12)*sharpe(Y') stats.tstat.t(1) min(Y') skewness(Y')];

% High complexity
tmp1    = nanmean(timing(:,end,end,:),4);
Yprdtmp1= nanmean(Yprd(:,end,end,:),4);
loc     = find(~isnan(Y'+Yprdtmp1));
r2tmp   = 1-nanvar(Y(loc)'-Yprdtmp1(loc))/nanvar(Y(loc)');
stats   = regstats(tmp1,ones(size(Y')),1,'tstat');
out     = [out ; r2tmp sqrt(12)*sharpe(tmp1) stats.tstat.t(1) min(tmp1) skewness(tmp1)];

% High complexity vs mkt
stats   = regstats(tmp1,Y','linear',{'tstat','rsquare','r'}); sqrt(12)*stats.tstat.beta(1)/nanstd(stats.r), stats.tstat.t(1)
tmp1res = stats.tstat.beta(1) + stats.r;
out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];

% GW 
tmp1    = timing_gy(:,1);
Yprd_gy_tmp1 = Yprd_gy(:,1);
loc     = find(~isnan(Y'+Yprd_gy_tmp1));
r2tmp   = 1-nanvar(Y(loc)'-Yprd_gy_tmp1(loc))/nanvar(Y(loc)');
stats   = regstats(tmp1,ones(size(Y')),1,'tstat');
out     = [out ; r2tmp sqrt(12)*sharpe(tmp1) stats.tstat.t(1) min(tmp1) skewness(tmp1)];

tmp1    = timing_gy(:,2);
Yprd_gy_tmp1 = Yprd_gy(:,2);
loc     = find(~isnan(Y'+Yprd_gy_tmp1));
r2tmp   = 1-nanvar(Y(loc)'-Yprd_gy_tmp1(loc))/nanvar(Y(loc)');
stats   = regstats(tmp1,ones(size(Y')),1,'tstat');
out     = [out ; r2tmp sqrt(12)*sharpe(tmp1) stats.tstat.t(1) min(tmp1) skewness(tmp1)];

tmp1    = timing_gy(:,end-2);  % z = 10
Yprd_gy_tmp1 = Yprd_gy(:,end-2);
loc     = find(~isnan(Y'+Yprd_gy_tmp1));
r2tmp   = 1-nanvar(Y(loc)'-Yprd_gy_tmp1(loc))/nanvar(Y(loc)');
stats   = regstats(tmp1,ones(size(Y')),1,'tstat');
out     = [out ; r2tmp sqrt(12)*sharpe(tmp1) stats.tstat.t(1) min(tmp1) skewness(tmp1)];

tmp1    = timing_gy(:,end);  % z = 1000
Yprd_gy_tmp1 = Yprd_gy(:,end);
loc     = find(~isnan(Y'+Yprd_gy_tmp1));
r2tmp   = 1-nanvar(Y(loc)'-Yprd_gy_tmp1(loc))/nanvar(Y(loc)');
stats   = regstats(tmp1,ones(size(Y')),1,'tstat');
out     = [out ; r2tmp sqrt(12)*sharpe(tmp1) stats.tstat.t(1) min(tmp1) skewness(tmp1)];

% GW vs mkt
stats   = regstats(tmp1,Y','linear',{'tstat','rsquare','r'}); sqrt(12)*stats.tstat.beta(1)/nanstd(stats.r), stats.tstat.t(1)
tmp1res = stats.tstat.beta(1) + stats.r;
out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];

% High complexity vs GW
tmp1    = nanmean(timing(:,end,end,:),4);
stats   = regstats(tmp1,timing_gy(:,end),'linear',{'tstat','rsquare','r'}); sqrt(12)*stats.tstat.beta(1)/nanstd(stats.r), stats.tstat.t(1)
tmp1res = stats.tstat.beta(1) + stats.r;
out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];

% GW ridgeless vs mkt
tmp1    = timing_gy(:,1);
stats   = regstats(tmp1,Y','linear',{'tstat','rsquare','r'}); sqrt(12)*stats.tstat.beta(1)/nanstd(stats.r), stats.tstat.t(1)
tmp1res = stats.tstat.beta(1) + stats.r;
out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];

% GW ridgeless vs GW with z = 1000
tmp1    = timing_gy(:,1);
stats   = regstats(tmp1,timing_gy(:,end),'linear',{'tstat','rsquare','r'}); sqrt(12)*stats.tstat.beta(1)/nanstd(stats.r), stats.tstat.t(1)
tmp1res = stats.tstat.beta(1) + stats.r;
out     = [out ; nan sqrt(12)*sharpe(tmp1res) stats.tstat.t(1) min(tmp1res) skewness(tmp1res)];

disp(['trwnin ' num2str(trnwin)])
disp([sqrt(12)*sharpe(tmp1res) stats.tstat.t(1)])

if saveon==1
    suffix      = ['trnwin-' num2str(trnwin) '-gamma-' gamma_str '-stdize-' num2str(stdize) '-demean-' num2str(demean) '-' num2str(subbeg(ss)) '-' num2str(subend(ss))];
    writematrix(out',[figdir suffix '-out.csv']);
end

toc
delete(gcp('nocreate'));

end