clear
tic

%**************************************************************************
% Parameters Setting
%**************************************************************************

% gamma in Random Fourier Features
gamma = 2;

% training window list
trnwin_list = [12, 60, 120];

% number of simulations
nSim = 1000;

% Standardization = True
stdize = 1;

%**************************************************************************
% Predictions for 1000 simulations with random seeds from 1 to 1000
% Note: Parallelization or HPC can be used to expedite the for loop
%**************************************************************************

for trnwin = trnwin_list
    for random_seed = 1:nSim
        % Each loop costs around 200 seconds
        tryrff_v2_function_for_each_sim(gamma, trnwin, random_seed, stdize);
    end
end
