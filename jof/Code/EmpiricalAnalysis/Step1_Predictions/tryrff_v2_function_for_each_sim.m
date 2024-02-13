function [] = tryrff_v2_function_for_each_sim(gamma, trnwin, iSim, stdize)

%**************************************************************************
% The function computes OOS performance with one random seed. 
% Parameters:
% gamma: gamma in Random Fourier Features
% trnwin: training window
% iSim: random seed for this simulation
% stdize: Standardization. stdize = 1 means True
%**************************************************************************

tic
nSim = 1; % total number of simulations run in this function

%**************************************************************************
% Choices
%**************************************************************************
% max number of Random Fourier Features (RFFs)
maxP    = 12000; 

% the grid of RFFs number
Plist   = [2 5:floor(trnwin/10):(trnwin-5) (trnwin-4):2:(trnwin+4) (trnwin+5):floor(trnwin/2):30*trnwin (31*trnwin):(10*trnwin):(maxP-1) maxP];

% training frequency
trainfrq= 1;

% shrinkage parameters lambda (z)
log_lamlist = [-3:1:3];
lamlist = 10.^(log_lamlist);

% save the result
saveon  = 1;

% Demeaning = False
demean  = 0;

% length of shrinkage parameters
nL      = length(lamlist);

% length of RFFs number grid
nP      = length(Plist);

% saving string
para_str = strcat('maxP-', num2str(maxP), '-trnwin-', num2str(trnwin), '-gamma-', num2str(gamma), '-stdize-', num2str(stdize), '-demean-', num2str(demean), '-v2');

%**************************************************************************
% Save path
%**************************************************************************
pwd_str = pwd; % get the local paths
save_path = strcat('./tryrff_v2_SeparateSims/', para_str);
mkdir(save_path); % build the saving path

%**************************************************************************
% Load Data
%**************************************************************************

load GYdata

% Y is the returns time series 
% X is the matrix of predictors (already lagged by 1 month)

% Add lag return (Y variable) as a predictor
X       = [X lagmatrix(Y,[1])];

% Vol-standardize
if stdize==1
    % Standardize X using expanding window
    X       = volstdbwd(X,[]);
    
    % Standardize Y (returns) by volatility of previous 12 months
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

T       = length(Y);
X       = X';
Y       = Y';
d       = size(X,1);

%**************************************************************************
% Output Space
%**************************************************************************

Yprd    = nan(T,nP,nL,nSim); % predicted Y
Bnrm    = nan(T,nP,nL,nSim); % beta norm

%**************************************************************************
% Recursive Estimation
%**************************************************************************

s = iSim;
disp(s);
disp(nP);
sStart = tic;

% Fix the random seed for random features
rng(s);

% Fix random features for maxP, then slice data
% W is the matrix of random Gaussian weights 
W       = randn(max(Plist),d);

for p=1:nP
    disp(p);
    
    P           = floor(Plist(p)/2);
    wtmp        = W(1:P,:);
    % only now do we build random Fourier features from raw features, X, 
    % and Gaussian weights, wtmp, and then applying cos and sin 
    Z           = [cos(gamma*wtmp*X);sin(gamma*wtmp*X)];
    Yprdtmp     = nan(T,nL);
    Bnrmtmp     = nan(T,nL);
    for t=trnwin+1:T
        
        % time-rolling window data processing
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
            % now we run the ridge regression 
            if P <= trnwin
                B       = ridgesvd(Ytrn',Ztrn',lamlist*trnwin);
            else
                % when P > trnwin , we use our own way of computing betas 
                B       = get_beta(Ytrn',Ztrn',lamlist);
            end         
        end
        
        % Test
        % this is our brediction: beta'*random_features + mean (if we did
        % subtract the mean from returns)
        Yprdtmp(t,:)= B'*Ztst + Ymn;
        Bnrmtmp(t,:) = sum(B.^2);
        
    end
    Yprd(:,p,:,1)   = Yprdtmp;
    Bnrm(:,p,:,1)   = Bnrmtmp;
end
sEnd = toc(sStart);

rntm    = toc;
if saveon==1    
    if iSim == 1
        save([save_path '/iSim' num2str(iSim) '.mat']);
    else 
        save([save_path '/iSim' num2str(iSim) '.mat'], 'Yprd', 'Bnrm');
    end
end

toc

end