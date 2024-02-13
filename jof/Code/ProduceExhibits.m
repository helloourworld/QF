% This is the Matlab code for paper:
% Kelly, Bryan T., Semyon Malamud, and Kangying Zhou. The virtue of complexity 
% in return prediction. No. w30217. National Bureau of Economic Research, 2022.

%**************************************************************************
% Preliminaries
%**************************************************************************
% This code is constructed in Matlab R2020b
clear all
clc

%**************************************************************************
% Calibrate Theory Model
%**************************************************************************

%%% Correctly Specified Model: Figure 1, Figure 2, Figure 3
% Note: First run Python code
% 'Calibration/CorrectlySpecifiedExhibits/correctly_specified_functions.py' 
% to get data in folder './Calibration/CorrectlySpecifiedExhibits/CorrectSpec_data'
run Calibration/CorrectlySpecifiedExhibits/rffexhibits_CorrectSpec.m

%%% Mis-Specified Model: Figure 4, Figure 5, Figure 6
% Note: First run Python code
% 'Calibration/MisSpecifiedExhibits/misspecified_functions_Psi_Identity.py' 
% to get data in folder './MisSpecifiedExhibits/MisSpec_data'
run Calibration/MisSpecifiedExhibits/rffexhibits_MisSpec.m

%**************************************************************************
% Empirical Analysis
%**************************************************************************

%%% Run OLS benchmark
run EmpiricalAnalysis/Step1_Predictions/GW_benchmark_main.m

%%% Run 1000 simulations
run EmpiricalAnalysis/Step1_Predictions/predictions_main.m

%%% Generate exhibits: Figure 7, Figure 8, Figure 9, Table I
% Figure F1, Figure F2, Figure F3, Figure F9, Figure F10, Figure F11
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_main.m

%%% Generate Figure 10
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_TimingPositions.m

%%% Generate Figure 11
% Run 1000 simulations
run EmpiricalAnalysis/Step1_Predictions/DropOnePredictor_main.m
% Generate Figure 11
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_DropOnePredictor.m

%**************************************************************************
% Appendix
%**************************************************************************

% Note: Figure F1, Figure F2, Figure F3, Figure F9, Figure F10, Figure F11 
% are generated in Empirical Analysis

%%% Generate Figure F4
run EmpiricalAnalysis/Step2_RFFexhibits/test_GW_vol.m

%%% Generate Figure F5
% Run 1000 simulations
run EmpiricalAnalysis/Step1_Predictions/variableimportance_main.m
% Generate Figure F5
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_NonlinearPredictionEffects.m

%%% Generate Table FI and Section G (Comparison With Momentum)
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_ComparisonWithMomentum.m

%%% Figure F6, Figure F7 and Figure F8
% Run 1000 simulations 
run EmpiricalAnalysis/Step1_Predictions/predictions_robustness.m
% Generate Figure F6, Figure F7 and Figure F8
run EmpiricalAnalysis/Step2_RFFexhibits/rffexhibits_robustness.m

