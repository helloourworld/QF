clear
tic

%**************************************************************************
% Parameters Setting
%**************************************************************************

% gamma in Random Fourier Features
gamma_list = [1, 0.5];

% training window
trnwin = 12;

% number of simulations
nSim = 1000;

% Standardization = True
stdize = 1;

%**************************************************************************
% Predictions for 1000 simulations with random seeds from 1 to 1000
% Note: Parallelization or HPC can be used to expedite the for loop
%**************************************************************************

for gamma = gamma_list
    for random_seed = 1:nSim
        tryrff_v2_function_for_each_sim(gamma, trnwin, random_seed, stdize);
    end
end

%**************************************************************************
% Predictions for 1000 simulations with random seeds from 1 to 1000
% with Standardization = False
%**************************************************************************

% gamma in Random Fourier Features
gamma = 2;

% training window
trnwin = 12;

% Standardization = False
stdize = 0;

% 1000 simulations
for random_seed = 1:nSim
    tryrff_v2_function_for_each_sim(gamma, trnwin, random_seed, stdize);
end
    
