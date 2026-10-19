%% 1. Loading the prepared data

% Loading the prepared rodent dataset
load("parasiteModel.mat", "parasiteModel");


%% 2. Extracting the required variables 

% Response variable
parasiteLoad = parasiteModel.Total_parasitical_load;

% Predictor variables
species = parasiteModel.SPECIES;
habitats = parasiteModel.HBT_TYPE;
sex = parasiteModel.SEX;
breeding = parasiteModel.BRS;
totalLength = parasiteModel.TL;

% Creating a table for statistical modelling
modelData = table(parasiteLoad, species, habitats, ...
    sex, breeding, totalLength);


%% 3. Fitting the Poisson Model here

poissonModel = fitglm(modelData, ...
    'parasiteLoad ~ species + habitats + sex + breeding + totalLength', ...
    'Distribution', 'poisson', ...
    'Link', 'log', ...
    'CategoricalVars', {'species', 'habitats', 'sex', 'breeding'});

% Displaying the model
disp(poissonModel);


%% 4. Dispersion of the Poisson Model

% Extracting Pearson residuals
pearsonResiduals = poissonModel.Residuals.Pearson;

% Calculating Pearson chi-square statistic
pearsonChiSquare = sum(pearsonResiduals.^2);

% Extracting residual degrees of freedom
residualDF = poissonModel.DFE;

% Calculating dispersion ratio
dispersionRatio = pearsonChiSquare / residualDF;

% Displaying results
fprintf('\n');
disp("-----------------------------------")
disp("Poisson Model Dispersion Assessment")
disp("-----------------------------------")
fprintf("Pearson chi-square: %.2f\n", pearsonChiSquare);
fprintf("Residual degrees of freedom: %d\n", residualDF);
fprintf("Dispersion ratio: %.3f\n", dispersionRatio);


%% 5. Investigating Poisson Model Residuals

% Extracting fitted parasite counts
fittedCounts = predict(poissonModel, modelData);

% Creating a residual plot
figure;
scatter(fittedCounts, pearsonResiduals, 15, 'filled');
xlabel('Fitted parasite count');
ylabel('Pearson residual');
title('Poisson Model: Pearson Residuals vs Fitted Counts');
grid on;

% Adding a horizontal reference line at zero
yline(0, '--');


%% Poisson Model Summary

% Number of observations and estimated coefficients
n = poissonModel.NumObservations;
k = height(poissonModel.Coefficients);

% Log-likelihood
logLikelihood = poissonModel.LogLikelihood;

% Information criteria
AIC = -2 * logLikelihood + 2 * k;
BIC = -2 * logLikelihood + k * log(n);

% Predicted parasite counts
fittedCounts = predict(poissonModel, modelData);

% Pearson residuals
pearsonResiduals = (parasiteLoad - fittedCounts) ...
    ./ sqrt(fittedCounts);

% Pearson chi-square statistic
pearsonChiSquare = sum(pearsonResiduals.^2);

% Residual degrees of freedom
residualDF = n - k;

% Dispersion ratio
dispersionRatio = pearsonChiSquare / residualDF;

% Display model summary
fprintf('\n');
disp('---------------------');
disp('POISSON MODEL SUMMARY');
disp('---------------------');

fprintf('Observations:             %d\n', n);
fprintf('Estimated parameters:     %d\n', k);
fprintf('Log-likelihood:           %.2f\n', logLikelihood);
fprintf('AIC:                      %.2f\n', AIC);
fprintf('BIC:                      %.2f\n', BIC);
fprintf('Pearson chi-square:       %.2f\n', pearsonChiSquare);
fprintf('Residual degrees freedom: %d\n', residualDF);
fprintf('Pearson dispersion ratio: %.3f\n', dispersionRatio);

