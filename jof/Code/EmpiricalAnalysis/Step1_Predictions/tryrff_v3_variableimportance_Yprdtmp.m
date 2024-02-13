function [] = tryrff_v3_variableimportance_Yprdtmp(gamma, trnwin, iSim)

%**************************************************************************
% The function computes variable importance
% Parameters:
% gamma: gamma in Random Fourier Features
% trnwin: training window
% iSim: random seed for this simulation
%**************************************************************************

%**************************************************************************
% Choices
%**************************************************************************
tic
P       = 12000; 
stdize  = 1;
lamlist = 10.^3;
saveon  = 1;
demean  = 0;
nSim    = 1;
trainfrq= 1;

para_str = strcat('maxP-', num2str(P), '-trnwin-', num2str(trnwin), ...
    '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', ...
    num2str(demean), '-variableimportance-Yprdtmp');

pwd_str = pwd; % get the local paths
save_path = strcat('./tryrff_v2_SeparateSims/', para_str);
mkdir(save_path); % build the saving path

plot_save_path = strcat(save_path, '/plots/');
mat_save_path = strcat(save_path, '/mat/');
mkdir(plot_save_path);
mkdir(mat_save_path);

%**************************************************************************
% Load Data
%**************************************************************************

load GYdata

% Add lag ret as a predictor
X       = [X lagmatrix(Y,[1])];

% Vol-standardize
if stdize==1
    % Standardize X
    X       = volstdbwd(X,[]);
    
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
Yprd    = nan(T,nSim);
Bnrm    = nan(T,nSim);
Ball    = nan(T,P,nSim);

%**************************************************************************
% Recursive Estimation
%**************************************************************************
% Fix random features
rng(iSim);
W           = randn(P/2,d);
Z           = [cos(gamma*W*X);sin(gamma*W*X)];
Yprdtmp     = nan(T,1);
Bnrmtmp     = nan(T,1);
Balltmp     = nan(T,P);
for t=trnwin+1:T
    trnloc  = (t-trnwin):t-1;
    Ztrn    = Z(:,trnloc);
    Ytrn    = Y(trnloc);
    Ztst    = Z(:,t);
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
    if t==trnwin+1 || mod(t-trnwin-1,trainfrq)==0
        
        if P <= trnwin
            B       = ridgesvd(Ytrn',Ztrn',lamlist*trnwin);
        else
            B       = get_beta(Ytrn',Ztrn',lamlist);
        end
        
    end

    % Test
    Yprdtmp(t,1) = B'*Ztst + Ymn;
    Bnrmtmp(t,1) = sum(B.^2);
    Balltmp(t,:) = B;
end
Yprd(:,1)   = Yprdtmp;
Bnrm(:,1)   = Bnrmtmp;
Ball(:,:)   = Balltmp;

%**************************************************************************
% Nonlinearity Sketch
%**************************************************************************

Yprdtmp     = nan(T,100,15);
for g=1:15
    
    xlist   = linspace(min(X(g,:)),max(X(g,:)),100);
    for i=1:100  % 100 steps between min and max
        Xtmp        = X;
        Xtmp(g,:)   = xlist(i);
        Z           = [cos(gamma*W*Xtmp);sin(gamma*W*Xtmp)];
        for t=trnwin+1:T
            Ztst    = Z(:,t);
            Zstd    = nanstd(Ztrn,[],2);
            Ztst    = Ztst./Zstd;

            % Test
            Yprdtmp(t,i,g)= Ball(t,:)*Ztst;
        end
    end
    g
end

if saveon==1
    
    % save mat
    if iSim == 1
        save([mat_save_path '/iSim' num2str(iSim) '.mat']);
    else 
        save([mat_save_path '/iSim' num2str(iSim) '.mat'], 'Yprdtmp');
    end
end

X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt'};
    
figure
set(gcf,'Position',[1 1 900 1200]);
for g=1:15
    xvec = linspace(-1,1,100)';
    yvec = nanmean(squeeze(Yprdtmp(:,:,g)));
    subplot(8,2,g)
    plot(xvec,yvec), set(gca,'ylim',[min(yvec),max(yvec)])
    hold on
    yline(0)
    title(X_columns(g))
%     pause
end

if saveon==1
    % save plot
    saveas(gcf,[plot_save_path '/iSim' num2str(iSim) '.eps'],'eps2c');
end

end