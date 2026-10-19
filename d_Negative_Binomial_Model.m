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


%% 3. Creating a Design Matrix for the Negative Binomial Model

% Fitting an initial Poisson model to obtain starting coefficient estimates 
poissonModel = fitglm(modelData, ...
    'parasiteLoad ~ species + habitats + sex + breeding + totalLength', ...
    'Distribution', 'poisson', ...
    'Link', 'log', ...
    'CategoricalVars', {'species', 'habitats', 'sex', 'breeding'});

% Changing the categorical predictors into numerical indicator variables
speciesDummy = dummyvar(categorical(modelData.species));
habitatDummy = dummyvar(categorical(modelData.habitats));
sexDummy = dummyvar(categorical(modelData.sex));
breedingDummy = dummyvar(categorical(modelData.breeding));

% Creating the intercept column
intercept = ones(height(modelData), 1);

% Combining the predictors into one numerical matrix
X = [intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Extracting observed parasite counts
Y = modelData.parasiteLoad;

% Extracting starting coefficient estimates from the Poisson model
initialBeta = poissonModel.Coefficients.Estimate;

% Checking the order of categorical variables
disp("Species categories:");
disp(categories(categorical(modelData.species)));
disp("Habitat categories:");
disp(categories(categorical(modelData.habitats)));
disp("Sex categories:");
disp(categories(categorical(modelData.sex)));
disp("Breeding categories:");
disp(categories(categorical(modelData.breeding)));

% Displaying the initial Poisson regression coefficient estimates 
fprintf('\n');
disp("----------------------------------------------");
disp("Poisson Regression Coefficient Estimates Table");
disp("----------------------------------------------");
disp(poissonModel.Coefficients);


%% 4. Creating the Negative Binomial parameters

% The initial estimate for the dispersion parameter
initialAlpha = 1;

% Making sure alpha remains positive
initialLogAlpha = log(initialAlpha);

% Combining the regression coefficients and dispersion parameter
initialParameters = [initialBeta; initialLogAlpha];

% Displaying the number of parameters to be estimated
fprintf("Number of negative binomial parameters: %d\n", ...
    numel(initialParameters));


%% 5. Fitting the Negative Binomial Model

% Creating the negative log-likelihood function
objectiveFunction = @(parameters) negativeBinomialNLL(parameters, X, Y);

% Setting optimisation options
options = optimset('Display', 'iter', ...
    'MaxFunEvals', 20000, ...
    'MaxIter', 20000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Estimating the negative binomial parameters
[estimatedParameters, negativeLogLikelihood, exitflag] = ...
    fminsearch(objectiveFunction, initialParameters, options);

%% 6. Extracting the Negative Binomial Parameter Estimates

% Extracting the estimated regression coefficients
estimatedBeta = estimatedParameters(1:end-1);

% Extracting the estimated dispersion parameter
estimatedAlpha = exp(estimatedParameters(end));

% Calculating the fitted parasite counts
predictedCounts = exp(X * estimatedBeta);

% Displaying the model estimates
fprintf('\n');
disp("-------------------------------------------");
disp("Negative Binomial Model Parameter Estimates");
disp("-------------------------------------------");

% Creating a coefficient estimates table
coefficientNames = poissonModel.CoefficientNames(:);

coefficientTable = table(coefficientNames, estimatedBeta, ...
    'VariableNames', {'Coefficient', 'Estimate'});

disp(coefficientTable);

% Displaying the dispersion parameter
fprintf("Estimated dispersion parameter (alpha): %.4f\n", ...
    estimatedAlpha);

% Displaying the log-likelihood
fprintf("Log-likelihood: %.2f\n", -negativeLogLikelihood);

% Displaying the optimisation exit flag
fprintf("Optimisation exit flag: %d\n", exitflag);


%% 7. Negative Binomial Dispersion Diagnostics

% Calculating the fitted variance under the negative binomial model
fittedVariance = predictedCounts + ...
    estimatedAlpha .* predictedCounts.^2;

% Calculating Pearson residuals
pearsonResiduals = (Y - predictedCounts) ./ sqrt(fittedVariance);

% Calculating the Pearson chi-square statistic
pearsonChiSquare = sum(pearsonResiduals.^2);

% Calculating residual degrees of freedom
residualDF = length(Y) - length(estimatedBeta);

% Calculating the Pearson dispersion ratio
dispersionRatio = pearsonChiSquare / residualDF;

% Displaying diagnostic results
fprintf('\n');
disp("----------------------------------------");
disp("Negative Binomial Dispersion Diagnostics");
disp("----------------------------------------");

fprintf("Pearson chi-square: %.2f\n", pearsonChiSquare);
fprintf("Residual degrees of freedom: %d\n", residualDF);
fprintf("Dispersion ratio: %.3f\n", dispersionRatio);


%% 8. Negative Binomial Residual Analysis

% Plotting Pearson residuals against fitted parasite counts
figure;

scatter(predictedCounts, pearsonResiduals, 15, 'filled');

% Adding a horizontal reference line
yline(0, '--');

% Labelling the graph
xlabel('Fitted parasite count');
ylabel('Pearson residual');
title('Negative Binomial Model: Pearson Residuals vs Fitted Counts');

grid on;


%% 9. Investigating Extreme Residuals

% Finding the observation with the largest Pearson residual
[maxResidual, maxIndex] = max(pearsonResiduals);

% Displaying its observed and predicted parasite counts
fprintf('\n');
disp("----------------------------------");
disp("Investigating the extreme residual");
disp("----------------------------------");
fprintf("Observation number: %d\n", maxIndex);
fprintf("Observed parasite count: %.0f\n", Y(maxIndex));
fprintf("Predicted parasite count: %.2f\n", predictedCounts(maxIndex));
fprintf("Pearson residual: %.2f\n", maxResidual);

% Displaying the corresponding rodent's information
disp(modelData(maxIndex, :));


%% 10. Checking Zero Counts in the Negative Binomial Model

% Counting the observed zero parasite loads
observedZeros = sum(Y == 0);

% Calculating the probability of zero parasites for each rodent
predictedZeroProbabilities = ...
    (1 + estimatedAlpha .* predictedCounts).^(-1 / estimatedAlpha);

% Calculating the expected total number of zeros
expectedZeros = sum(predictedZeroProbabilities);

% Displaying the results
fprintf('\n');
disp("----------------------------------------");
disp("Negative Binomial Zero Count Diagnostics");
disp("----------------------------------------");

fprintf("Observed zeros: %d\n", observedZeros);
fprintf("Expected zeros: %.2f\n", expectedZeros);
fprintf("Observed zero percentage: %.2f%%\n", ...
    100 * observedZeros / length(Y));
fprintf("Expected zero percentage: %.2f%%\n", ...
    100 * expectedZeros / length(Y));



%% 11. Simulating Zero Counts Under the Negative Binomial Model

% Setting a random seed to ensure reproducible simulation results
rng(42);

% Setting the number of simulations
numSimulations = 10000;

% Initialising a vector to store the simulated zero counts
simulatedZeros = zeros(numSimulations, 1);

% Setting the negative binomial size parameter
r = 1 / estimatedAlpha;

% Simulating parasite counts under the fitted model
for i = 1:numSimulations

    % Converting the predicted means into NB probability parameters
    p = r ./ (r + predictedCounts);

    % Generating simulated parasite counts
    simulatedCounts = nbinrnd(r, p);

    % Counting the zeros in each simulated dataset
    simulatedZeros(i) = sum(simulatedCounts == 0);

end

% Calculating the proportion of simulations with at least 39 zeros
excessZeroProbability = mean(simulatedZeros >= observedZeros);

% Displaying the simulation results
fprintf('\n');
disp("---------------------------------------");
disp("Negative Binomial Zero Count Simulation");
disp("---------------------------------------");

fprintf("Observed zeros: %d\n", observedZeros);
fprintf("Mean simulated zeros: %.2f\n", mean(simulatedZeros));
fprintf("Probability of at least %d zeros: %.4f\n", ...
    observedZeros, excessZeroProbability);


%% 12. Negative Binomial Model Summary

% Calculating the number of estimated parameters
numParameters = numel(estimatedParameters);

% Calculating model selection criteria
AIC = 2 * numParameters + 2 * negativeLogLikelihood;
BIC = log(length(Y)) * numParameters + 2 * negativeLogLikelihood;

% Displaying the final model summary
fprintf('\n');
disp("-------------------------------");
disp("Negative Binomial Model Summary");
disp("-------------------------------");

fprintf("Number of observations: %d\n", length(Y));
fprintf("Number of estimated parameters: %d\n", numParameters);
fprintf("Log-likelihood: %.2f\n", -negativeLogLikelihood);
fprintf("AIC: %.2f\n", AIC);
fprintf("BIC: %.2f\n", BIC);
fprintf("Dispersion parameter (alpha): %.4f\n", estimatedAlpha);
fprintf("Pearson dispersion ratio: %.3f\n", dispersionRatio);
fprintf("Observed zero counts: %d\n", observedZeros);
fprintf("Expected zero counts: %.2f\n", expectedZeros);
fprintf("Optimisation exit flag: %d\n", exitflag);


%% 14. Creating the Negative Binomial - Negative Log-Likelihood Function

function nLL = negativeBinomialNLL(parameters, X, Y)

    % Extracting the regression coefficients
    beta = parameters(1:end-1);

    % Extracting and transforming the dispersion parameter
    alpha = exp(parameters(end));

    % Calculating the expected parasite counts
    mu = exp(X * beta);

    % Converting dispersion to the negative binomial size parameter
    r = 1 / alpha;

    % Calculating the negative binomial log-likelihood
    logLikelihood = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* log(r ./ (r + mu)) ...
        + Y .* log(mu ./ (r + mu));

    % Summing and negating the log-likelihood
    nLL = -sum(logLikelihood);

end