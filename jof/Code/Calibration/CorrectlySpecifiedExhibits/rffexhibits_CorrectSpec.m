clear all
clc
tic

%**************************************************************************
% Choices
%**************************************************************************
pwd_str = pwd; % get the local paths
data_path = './CorrectSpec_data/'
figdir = './CorrectSpec_exhibits/';
mkdir(figdir);

newcolors = {'#000000', '#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
use_newcolors = 1;
if use_newcolors
    colororder(newcolors);
end

%**************************************************************************
% Parameter settings and data loading
%**************************************************************************
lamlist = [0.01, 0.1, 1, 10, 50];
R2 = table2array(readtable(strcat(data_path, 'R2.csv')));
ER = table2array(readtable(strcat(data_path, 'ER.csv')));
vol = table2array(readtable(strcat(data_path, 'Vol.csv')));
SR = table2array(readtable(strcat(data_path, 'SR.csv')));
MSE = table2array(readtable(strcat(data_path, 'MSE.csv')));
Bnrmbar = table2array(readtable(strcat(data_path, 'Bnorm.csv')));

True = table2array(readtable(strcat(data_path, 'TRUE_Rsq_ET_Vol_SR.csv')));
R2_TRUE = True(1);
ER_TRUE = True(2);
Vol_TRUE = True(3);
SR_TRUE = True(4);

Plist = 1:1000;
trnwin = 100;

%**************************************************************************
% Plot
%**************************************************************************
% R2
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin,R2,'linewidth',1.5); yline(R2_TRUE,'--','color','red','linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'True'};{'$c=1$'}],'interpreter','latex','location','northeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.3001,.3]);
ytickformat('%.1f')
saveas(gcf,[figdir 'CorrectSpecR2.eps'],'eps2c');
close 'all'

% SR
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin, SR,'linewidth',1.5), yline(SR_TRUE,'--','color','red','linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'True'};{'$c=1$'}],'interpreter','latex','location','northeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20)
ytickformat('%.2f')
saveas(gcf,[figdir 'CorrectSpecSR.eps'],'eps2c');
close 'all'

% ER
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin,ER,'linewidth',1.5), yline(ER_TRUE,'--','color','red','linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'True'};{'$c=1$'}],'interpreter','latex','location','northeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0, 0.21])
ytickformat('%.2f')
saveas(gcf,[figdir 'CorrectSpecER.eps'],'eps2c');
close 'all'

% Vol
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin, vol,'linewidth',1.5), yline(Vol_TRUE,'--','color','red','linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,6])
ytickformat('%.2f')
saveas(gcf,[figdir 'CorrectSpecVol.eps'],'eps2c');
close 'all'

% MSR
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin,MSE,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0.5,6])
ytickformat('%.2f')
saveas(gcf,[figdir 'CorrectSpecMSE.eps'],'eps2c');
close 'all'

% Bnorm
if use_newcolors
    colororder(newcolors);
end
plot(Plist/trnwin,Bnrmbar,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$c$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,6])
ytickformat('%.2f')
saveas(gcf,[figdir 'CorrectSpecBnorm.eps'],'eps2c');
close 'all'

