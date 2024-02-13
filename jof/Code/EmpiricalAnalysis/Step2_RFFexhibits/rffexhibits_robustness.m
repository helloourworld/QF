clear all
clc

%**************************************************************************
% Parameters Setting
%**************************************************************************

% gamma in Random Fourier Features
gamma_list = [1, 0.5];

% training window
trnwin = 12;

% Standardization = True
stdize = 1;

%**************************************************************************
% Run function rffexhibits_function(gamma, trnwin, stdize) with
% gamma_list = [1, 0.5];
%**************************************************************************

for gamma = gamma_list
    rffexhibits_function(gamma, trnwin, stdize);
end

%**************************************************************************
% Run function rffexhibits_function(gamma, trnwin, stdize) with
% with Standardization = False
%**************************************************************************

% gamma in Random Fourier Features
gamma = 2;

% training window
trnwin = 12;

% Standardization = False
stdize = 0;

% generate exhibits
rffexhibits_function(gamma, trnwin, stdize);
    