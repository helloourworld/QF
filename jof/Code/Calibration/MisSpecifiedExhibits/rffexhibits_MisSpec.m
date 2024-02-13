%**************************************************************************
% Choices
%**************************************************************************
clear all
clc
tic

% spectual color
defaultcolors = {'#0072BD', '#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
newcolors = {'#000000', '#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
use_newcolors = 1;

pwd_str = pwd; % get the local paths
data_path = './MisSpec_data/';
figdir = './MisSpec_exhibits/';
mkdir(figdir);

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
plot(Plist/trnwin,R2,'linewidth',1.5); xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'$cq=1$'}],'interpreter','latex','location','northeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[-.3,.3]);
ytickformat('%.1f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecR2.eps'],'eps2c');
close 'all'

% SR
plot(Plist/trnwin, SR,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'$cq=1$'}],'interpreter','latex','location','southeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20)
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecSR.eps'],'eps2c');
close 'all'

% ER
plot(Plist/trnwin,ER,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
legend([{'Ridgeless'}; cellstr(strcat(strcat('$z=',num2str(lamlist')),'$'));{'$cq=1$'}],'interpreter','latex','location','northeast')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20);
set(gca,'ytick',[0, 0.01, 0.02, 0.03]);
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecER.eps'],'eps2c');
close 'all'

% Vol
plot(Plist/trnwin, vol,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,6])
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecVol.eps'],'eps2c');
close 'all'

% MSE
plot(Plist/trnwin,MSE,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0.5,6])
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecMSE.eps'],'eps2c');
close 'all'

% Bnorm
plot(Plist/trnwin,Bnrmbar,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,6])
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecBnorm.eps'],'eps2c');
close 'all'


% Bnorm times P
plot(Plist/trnwin,Plist.*Bnrmbar,'linewidth',1.5), xline(1,'--','color',.5*ones(1,3)); xlabel('$cq$','interpreter','latex')
set(gcf,'Position',[321 241 512 384]);
set(gca,'fontname','TimesNewRoman','fontsize',20,'ylim',[0,6])
ytickformat('%.2f')
if use_newcolors == 1
    colororder(newcolors)
end
saveas(gcf,[figdir 'MisSpecBnormTimesP.eps'],'eps2c');
close 'all'













    



