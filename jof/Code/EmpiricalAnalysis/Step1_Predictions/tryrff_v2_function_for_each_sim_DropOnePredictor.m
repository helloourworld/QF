function [] = tryrff_v2_function_for_each_sim_DropOnePredictor(gamma, trnwin, iSim)

%**************************************************************************
% The function computes OOS performance with one random seed and dropping 
% 1 variable at a time.
% Parameters:
% gamma: gamma in Random Fourier Features
% trnwin: training window
% iSim: random seed for this simulation
%**************************************************************************

tic
nSim = 1;

for iGW = 1:15 % iGW is the index of variable
    %**************************************************************************
    % Choices
    %**************************************************************************
    maxP    = 12000;
    Plist   = [2 5:floor(trnwin/10):(trnwin-5) (trnwin-4):2:(trnwin+4) (trnwin+5):floor(trnwin/2):30*trnwin (31*trnwin):(10*trnwin):(maxP-1) maxP];
        
    trainfrq= 1;
    stdize  = 1;
    log_lamlist = [-3:1:3];
    lamlist = 10.^(log_lamlist);
    saveon  = 1;
    demean  = 0;
    nL      = length(lamlist);
    nP      = length(Plist);
    para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-DropOneVariable');
    X_columns = {'dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr',...
        'dp', 'dy', 'ltr', 'ep', 'bm', 'ntis', 'lag-mkt'};
    
    %**************************************************************************
    % Save path
    %**************************************************************************
    pwd_str = pwd; % get the local paths
    save_path = strcat('./tryrff_v2_SeparateSims/', para_str, '/', ...
        num2str(iGW),'_', char(X_columns(iGW)));
    mkdir(save_path); % build the saving path
    
    %**************************************************************************
    % Load Data
    %**************************************************************************
    load GYdata
    
    % Add lag ret as a predictor
    % X       = [X lag(Y,1,0)];
    X       = [X lagmatrix(Y,[1])];
    
    % Vol-standardize
    if stdize==1
        % Standardize X
        X       = volstdbwd(X,[]);
        
        % Standardize Y by volatility of previous 12 months
        Y2      = 0;
        for j=1:12
            Y2  = Y2+lagmatrix(Y.^2,[j]);
        end
        Y2      = Y2/12; % Y2 is the moving average of previous 12 months
        Y       = Y./sqrt(Y2);
        clear Y2
        
        % Drop first 3 years due to vol scaling of X
        Y       = Y(37:end);
        X       = X(37:end,:);
        dates   = dates(37:end,:);
    end
    
    % Drop iGW predictor
    X(:, iGW) = [];
    
    T       = length(Y);
    X       = X';
    Y       = Y';
    d       = size(X,1);
    
    %**************************************************************************
    % Output Space
    %**************************************************************************
    
    Yprd    = nan(T,nL); % predicted Y
    Bnrm    = nan(T,nL); % beta norm
    Ball    = nan(T,maxP,nL);
    
    %**************************************************************************
    % Recursive Estimation
    %**************************************************************************
    s = iSim;
    disp(s);
    disp(nP);
    sStart = tic;
    
    % Fix random features
    rng(s);
    % Fix random features for maxP, then slice data
    W       = randn(max(Plist),d);
    
    % for p=1:nP
    p = nP;
    disp(p);
    
    P           = floor(Plist(p)/2);
    wtmp        = W(1:P,:);
    Z           = [cos(gamma*wtmp*X);sin(gamma*wtmp*X)];
    Yprdtmp     = nan(T,nL);
    Bnrmtmp     = nan(T,nL);
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
        Yprdtmp(t,:)= B'*Ztst + Ymn;
        Bnrmtmp(t,:) = sum(B.^2);
        
        % save beta
        Ball(t,:,:)= B;
    end
    
    Yprd   = Yprdtmp;
    Bnrm   = Bnrmtmp;
    
    sEnd = toc(sStart);
    
    rntm    = toc;
    if saveon==1
        
        if iSim == 1
            save([save_path '/iSim' num2str(iSim) '.mat']);
        else
            save([save_path '/iSim' num2str(iSim) '.mat'], 'Yprd');
        end
    end
    
end
toc

end