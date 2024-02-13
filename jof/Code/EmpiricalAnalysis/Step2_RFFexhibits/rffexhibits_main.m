clear all
clc

%**************************************************************************
% Parameters Setting
%**************************************************************************

% gamma in Random Fourier Features
gamma = 2;

% Choose training windows
training_window_list = [12, 60, 120];

% Standardization = True
stdize = 1;

%**************************************************************************
% Run function rffexhibits_function(gamma, trnwin, stdize)
%**************************************************************************

for trnwin = training_window_list
    rffexhibits_function(gamma, trnwin, stdize);
end

