%% 1. Loading the prepared data

% Loading the prepared rodent dataset
load("parasiteModel.mat", "parasiteModel");


%% 2. Extracting the required variables 

% Response variable
parasiteLoad = parasiteModel.Total_parasitical_load;

% Extracting the explanatory variables
species = categorical(parasiteModel.SPECIES);
habitats = categorical(parasiteModel.HBT_TYPE);
sex = categorical(parasiteModel.SEX);
breeding = categorical(parasiteModel.BRS);
totalLength = parasiteModel.TL;

% Creating the model data table
modelData = table( ...
    parasiteLoad, species, habitats, sex, breeding, totalLength);

% Displaying the number of observations
fprintf("Number of observations: %d\n", height(modelData));


%% 3. Constructing the Design Matrices

% Creating dummy variables for categorical predictors
speciesDummy = dummyvar(modelData.species);
habitatDummy = dummyvar(modelData.habitats);
sexDummy = dummyvar(modelData.sex);
breedingDummy = dummyvar(modelData.breeding);

% Creating the intercept
intercept = ones(height(modelData), 1);

% Extracting the response variable
Y = modelData.parasiteLoad;

% Full negative binomial model
Xfull = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Model excluding habitat
XnoHabitat = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Model excluding species
XnoSpecies = [ ...
    intercept, ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Checking the number of parameters in each design matrix
fprintf("Full model coefficients: %d\n", size(Xfull, 2));
fprintf("Without habitat coefficients: %d\n", size(XnoHabitat, 2));
fprintf("Without species coefficients: %d\n", size(XnoSpecies, 2));

% Checking for redundant columns
assert(rank(Xfull) == size(Xfull, 2), ...
    "The full design matrix is not full rank.");
assert(rank(XnoHabitat) == size(XnoHabitat, 2), ...
    "The no-habitat design matrix is not full rank.");
assert(rank(XnoSpecies) == size(XnoSpecies, 2), ...
    "The no-species design matrix is not full rank.");

%% 4. Fitting the Negative Binomial Models

% Storing the three design matrices into a bigger design matrix 
designMatrices = {Xfull, XnoHabitat, XnoSpecies};

% Defining the model names
modelNames = [ ...
    "Full NB";
    "NB Without Habitat";
    "NB Without Species"];

% Counting the models
numModels = numel(designMatrices);

% Initialising vectors for model results
logLikelihoods = zeros(numModels, 1);
AICvalues = zeros(numModels, 1);
BICvalues = zeros(numModels, 1);
numParameters = zeros(numModels, 1);
exitFlags = zeros(numModels, 1);

% Initialising storage for fitted parameters
fittedParameters = cell(numModels, 1);

% Setting optimisation options
options = optimset( ...
    'Display', 'off', ...
    'MaxFunEvals', 20000, ...
    'MaxIter', 20000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Fitting each model
for i = 1:numModels

    % Selecting the current design matrix
    Xcurrent = designMatrices{i};

    % Obtaining initial regression coefficients
    initialBeta = Xcurrent \ log(Y + 0.5);

    % Initialising the log-dispersion parameter
    initialLogAlpha = log(1);

    % Combining initial parameter estimates
    initialParameters = [initialBeta; initialLogAlpha];

    % Defining the negative binomial objective function
    objectiveFunction = @(parameters) ...
        negativeBinomialNLL(parameters, Xcurrent, Y);

    % Estimating the model parameters
    [estimatedParameters, negativeLogLikelihood, exitflag] = ...
        fminsearch(objectiveFunction, initialParameters, options);

    % Checking whether the model converged
if exitflag ~= 1 || ~isfinite(negativeLogLikelihood)
    error("Negative binomial model %d did not converge.", i);
end

    % Storing fitted parameters
    fittedParameters{i} = estimatedParameters;

    % Storing the optimisation exit flag
    exitFlags(i) = exitflag;

    % Calculating the maximised log-likelihood
    logLikelihoods(i) = -negativeLogLikelihood;

    % Counting regression and dispersion parameters
    numParameters(i) = numel(estimatedParameters);

    % Calculating AIC
    AICvalues(i) = ...
        2 * numParameters(i) - 2 * logLikelihoods(i);

    % Calculating BIC
    BICvalues(i) = ...
        log(length(Y)) * numParameters(i) ...
        - 2 * logLikelihoods(i);
end


%% 5. Comparing the Negative Binomial Models

% Creating a table of model results
modelComparisonTable = table( ...
    modelNames, ...
    numParameters, ...
    logLikelihoods, ...
    AICvalues, ...
    BICvalues, ...
    exitFlags, ...
    'VariableNames', ...
    {'Model', 'Parameters', 'LogLikelihood', ...
     'AIC', 'BIC', 'ExitFlag'});

% Displaying the comparison table
fprintf('\n');
disp("----------------------------------------------");
disp("Negative Binomial Variable Importance Analysis");
disp("----------------------------------------------");
disp(modelComparisonTable);


%% 5.1 Calculating Likelihood-Ratio Statistics

% Calculating the likelihood-ratio statistic for habitat
LRhabitat = 2 * (logLikelihoods(1) - logLikelihoods(2));

% Calculating the likelihood-ratio statistic for species
LRspecies = 2 * (logLikelihoods(1) - logLikelihoods(3));

% Calculating changes in AIC
deltaAIChabitat = AICvalues(2) - AICvalues(1);
deltaAICspecies = AICvalues(3) - AICvalues(1);

% Calculating changes in BIC
deltaBIChabitat = BICvalues(2) - BICvalues(1);
deltaBICspecies = BICvalues(3) - BICvalues(1);

% Displaying the results
fprintf("\nHabitat contribution:\n");
fprintf("Likelihood-ratio statistic: %.4f\n", LRhabitat);
fprintf("Change in AIC: %.4f\n", deltaAIChabitat);
fprintf("Change in BIC: %.4f\n", deltaBIChabitat);

fprintf("\nSpecies contribution:\n");
fprintf("Likelihood-ratio statistic: %.4f\n", LRspecies);
fprintf("Change in AIC: %.4f\n", deltaAICspecies);
fprintf("Change in BIC: %.4f\n", deltaBICspecies);


%% 5.2 Calculating Likelihood-Ratio Test P-Values

% Each reduced model removes three categorical coefficients
degreesOfFreedom = 3;

% Calculating approximate chi-squared p-values
pHabitat = chi2cdf(LRhabitat, degreesOfFreedom, 'upper');
pSpecies = chi2cdf(LRspecies, degreesOfFreedom, 'upper');

fprintf("\nHabitat likelihood-ratio p-value: %.6g\n", pHabitat);
fprintf("Species likelihood-ratio p-value: %.6g\n", pSpecies);



%% 6. Bootstrap Simulation of Variable Importance

% Setting the random seed for reproducibility
rng(42);

% Setting the number of bootstrap simulations
numBootstraps = 500;

% Obtaining the number of observations
numObservations = length(Y);

% Initialising vectors to store bootstrap results
bootstrapLRhabitat = NaN(numBootstraps, 1);
bootstrapLRspecies = NaN(numBootstraps, 1);
bootstrapDifference = NaN(numBootstraps, 1);

% Recording whether each bootstrap iteration is successful
bootstrapSuccess = false(numBootstraps, 1);

% Initialising counters for failed fits
failedOptimisations = 0;
rankDeficientSamples = 0;

% Repeating the simulation
for b = 1:numBootstraps

    % Sampling 474 row indices with replacement
    sampleIndices = randi(numObservations, numObservations, 1);

    % Extracting the bootstrap response variable
    Ybootstrap = Y(sampleIndices);

    % Extracting the bootstrap design matrices
    XfullBootstrap = Xfull(sampleIndices, :);
    XnoHabitatBootstrap = XnoHabitat(sampleIndices, :);
    XnoSpeciesBootstrap = XnoSpecies(sampleIndices, :);

    % Checking that all design matrices retain full column rank
    if rank(XfullBootstrap) < size(XfullBootstrap, 2) || ...
       rank(XnoHabitatBootstrap) < size(XnoHabitatBootstrap, 2) || ...
       rank(XnoSpeciesBootstrap) < size(XnoSpeciesBootstrap, 2)
        rankDeficientSamples = rankDeficientSamples + 1;
        continue;
    end

    % Storing the bootstrap design matrices
    bootstrapMatrices = { ...
        XfullBootstrap, ...
        XnoHabitatBootstrap, ...
        XnoSpeciesBootstrap};

    % Initialising model log-likelihoods
    bootstrapLogLikelihoods = NaN(3, 1);

    % Initialising a convergence indicator
    allModelsConverged = true;

    % Fitting the three models to the bootstrap sample
    for j = 1:3

        % Selecting the current design matrix
        Xcurrent = bootstrapMatrices{j};

        % Using the original fitted parameters as starting values
        initialParameters = fittedParameters{j};

        % Defining the negative binomial objective function
        objectiveFunction = @(parameters) ...
            negativeBinomialNLL(parameters, Xcurrent, Ybootstrap);

        % Fitting the negative binomial model
        [estimatedParameters, negativeLogLikelihood, exitflag] = ...
            fminsearch(objectiveFunction, initialParameters, options);

        % Checking convergence and finite likelihood
        if exitflag <= 0 || ~isfinite(negativeLogLikelihood)
            allModelsConverged = false;
            break;
        end

        % Storing the fitted log-likelihood
        bootstrapLogLikelihoods(j) = -negativeLogLikelihood;
    end

    % Skipping the sample if any model failed to converge
    if ~allModelsConverged
        failedOptimisations = failedOptimisations + 1;
        continue;
    end

    % Calculating habitat's likelihood-ratio contribution
    bootstrapLRhabitat(b) = 2 * ...
        (bootstrapLogLikelihoods(1) - bootstrapLogLikelihoods(2));

    % Calculating species' likelihood-ratio contribution
    bootstrapLRspecies(b) = 2 * ...
        (bootstrapLogLikelihoods(1) - bootstrapLogLikelihoods(3));

    % Calculating the difference in contributions
    bootstrapDifference(b) = ...
        bootstrapLRhabitat(b) - bootstrapLRspecies(b);

    % Recording a successful iteration
    bootstrapSuccess(b) = true;

    % Displaying progress every 50 iterations
    if mod(b, 50) == 0
        fprintf("Completed %d of %d bootstrap samples\n", ...
            b, numBootstraps);
    end
end


%% 7. Summarising the Bootstrap Results

% Counting successful bootstrap iterations
numSuccessful = sum(bootstrapSuccess);

% Extracting successful results
validHabitatLR = bootstrapLRhabitat(bootstrapSuccess);
validSpeciesLR = bootstrapLRspecies(bootstrapSuccess);
validDifference = bootstrapDifference(bootstrapSuccess);

% Checking that enough simulations succeeded
if numSuccessful == 0
    error("No bootstrap simulations completed successfully.");
end

% Calculating the average likelihood-ratio contributions
meanHabitatLR = mean(validHabitatLR);
meanSpeciesLR = mean(validSpeciesLR);

% Calculating the average difference
meanDifference = mean(validDifference);

% Calculating the percentage of successful samples
% where habitat contributes more than species
percentageHabitatGreater = ...
    100 * mean(validHabitatLR > validSpeciesLR);

% Calculating percentile intervals for the difference
differenceInterval = prctile(validDifference, [2.5, 97.5]);

% Displaying simulation results
fprintf('\n');
disp("-------------------------------------");
disp("Bootstrap Variable Importance Results");
disp("-------------------------------------");
fprintf("Requested bootstrap samples: %d\n", numBootstraps);
fprintf("Successful samples: %d\n", numSuccessful);
fprintf("Rank-deficient samples: %d\n", rankDeficientSamples);
fprintf("Failed optimisations: %d\n", failedOptimisations);
fprintf("\nMean habitat LR: %.4f\n", meanHabitatLR);
fprintf("Mean species LR: %.4f\n", meanSpeciesLR);
fprintf("Mean LR difference: %.4f\n", meanDifference);
fprintf("Habitat contribution greater than species: %.2f%%\n", ...
    percentageHabitatGreater);
fprintf("Bootstrap percentile interval for LR difference: [%.4f, %.4f]\n", ...
    differenceInterval(1), differenceInterval(2));


%% 8. Visualising the Bootstrap Results

% Creating a figure
figure;

% Plotting habitat likelihood-ratio contributions
histogram(validHabitatLR, ...
    'Normalization', 'probability', ...
    'FaceAlpha', 0.5);
hold on;

% Plotting species likelihood-ratio contributions
histogram(validSpeciesLR, ...
    'Normalization', 'probability', ...
    'FaceAlpha', 0.5);
xlabel('Likelihood-Ratio Contribution');
ylabel('Proportion of Bootstrap Samples');
title('Bootstrap Comparison of Habitat and Species');
legend('Habitat', 'Species');
grid on;
hold off;


%% 8.1 Visualising the Difference in Variable Importance

% Plotting the bootstrap LR differences
figure;
histogram(validDifference, 30);
hold on;

% Adding a reference line at zero
xline(0, '--', 'Equal Contributions');

% Adding the observed LR difference
observedDifference = LRhabitat - LRspecies;
xline(observedDifference, '-', 'Observed Difference');
xlabel('Habitat LR - Species LR');
ylabel('Number of Bootstrap Samples');
title('Bootstrap Difference in Variable Importance');
grid on;
hold off;



%% 9. Identifying Potentially Influential Observations

% Extracting the full NB regression coefficients
fullBeta = fittedParameters{1}(1:end-1);

% Extracting the full NB dispersion parameter
fullAlpha = exp(fittedParameters{1}(end));

% Calculating the predicted mean parasite load
predictedMean = exp(Xfull * fullBeta);

% Calculating the predicted negative binomial variance
predictedVariance = predictedMean + ...
    fullAlpha * predictedMean.^2;

% Calculating Pearson residuals
pearsonResiduals = ...
    (Y - predictedMean) ./ sqrt(predictedVariance);

% Sorting observations by absolute Pearson residual magnitude
[sortedResiduals, sortedIndices] = ...
    sort(abs(pearsonResiduals), 'descend');

% Selecting the 10 largest absolute residuals
numOutliers = min(10, length(Y));

% Extracting the indices of those observations
outlierIndices = sortedIndices(1:numOutliers);

% Creating a table of potentially influential observations
outlierTable = table( ...
    outlierIndices, ...
    Y(outlierIndices), ...
    predictedMean(outlierIndices), ...
    pearsonResiduals(outlierIndices), ...
    'VariableNames', ...
    {'RowIndex', 'ObservedCount', ...
     'PredictedCount', 'PearsonResidual'});

% Displaying the results
fprintf('\n');
disp("----------------------------------------------");
disp("Potentially Influential Observations");
disp("----------------------------------------------");
disp(outlierTable);


%% 10. Leave-One-Out Sensitivity Analysis

% Initialising vectors to store likelihood-ratio contributions
sensitivityLRhabitat = NaN(numOutliers, 1);
sensitivityLRspecies = NaN(numOutliers, 1);

% Initialising a vector for successful fits
sensitivitySuccess = false(numOutliers, 1);

% Repeating the analysis for each potentially influential rodent
for i = 1:numOutliers

    % Identifying the observation to temporarily exclude
    excludedIndex = outlierIndices(i);

    % Creating an index of retained observations
    retainedRows = true(length(Y), 1);

    retainedRows(excludedIndex) = false;

    % Creating the reduced response vector
    Ysensitivity = Y(retainedRows);

    % Creating the reduced design matrices
    XfullSensitivity = Xfull(retainedRows, :);
    XnoHabitatSensitivity = XnoHabitat(retainedRows, :);
    XnoSpeciesSensitivity = XnoSpecies(retainedRows, :);

    % Storing the three design matrices
    sensitivityMatrices = { ...
        XfullSensitivity, ...
        XnoHabitatSensitivity, ...
        XnoSpeciesSensitivity};

    % Initialising the model log-likelihoods
    sensitivityLogLikelihoods = NaN(3, 1);

    % Initialising the convergence indicator
    allModelsConverged = true;

    % Fitting the three models without the selected observation
    for j = 1:3

        % Selecting the current design matrix
        Xcurrent = sensitivityMatrices{j};

        % Using the original fitted parameters as starting values
        initialParameters = fittedParameters{j};

        % Defining the negative binomial objective function
        objectiveFunction = @(parameters) ...
            negativeBinomialNLL(parameters, Xcurrent, Ysensitivity);

        % Fitting the model
        [estimatedParameters, negativeLogLikelihood, exitflag] = ...
            fminsearch(objectiveFunction, initialParameters, options);

        % Checking convergence
        if exitflag <= 0 || ~isfinite(negativeLogLikelihood)
            allModelsConverged = false;
            break;
        end

        % Storing the fitted log-likelihood
        sensitivityLogLikelihoods(j) = -negativeLogLikelihood;
    end

    % Skipping failed model fits
    if ~allModelsConverged
        continue;
    end

    % Calculating the habitat likelihood-ratio contribution
    sensitivityLRhabitat(i) = 2 * ...
        (sensitivityLogLikelihoods(1) ...
        - sensitivityLogLikelihoods(2));

    % Calculating the species likelihood-ratio contribution
    sensitivityLRspecies(i) = 2 * ...
        (sensitivityLogLikelihoods(1) ...
        - sensitivityLogLikelihoods(3));

    % Recording a successful sensitivity analysis
    sensitivitySuccess(i) = true;
end


%% 11. Summarising the Sensitivity Analysis

% Calculating the original difference in LR contributions
originalDifference = LRhabitat - LRspecies;

% Calculating the sensitivity differences
sensitivityDifference = ...
    sensitivityLRhabitat - sensitivityLRspecies;

% Calculating changes relative to the original result
changeInDifference = ...
    sensitivityDifference - originalDifference;

% Creating a table of sensitivity results
sensitivityTable = table( ...
    outlierIndices, ...
    Y(outlierIndices), ...
    sensitivityLRhabitat, ...
    sensitivityLRspecies, ...
    sensitivityDifference, ...
    changeInDifference, ...
    sensitivitySuccess, ...
    'VariableNames', ...
    {'ExcludedRow', 'ParasiteCount', ...
     'HabitatLR', 'SpeciesLR', ...
     'LRDifference', 'ChangeFromOriginal', ...
     'SuccessfulFit'});

% Displaying results
fprintf('\n');
disp("----------------------------------------------");
disp("Leave-One-Out Sensitivity Results");
disp("----------------------------------------------");
disp(sensitivityTable);


%% 11.1 Evaluating the Stability of Habitat Importance

% Identifying successful sensitivity analyses
validResults = sensitivitySuccess;

% Calculating the percentage where habitat remains more important
if any(validResults)
    habitatStillGreater = ...
        100 * mean(sensitivityDifference(validResults) > 0);
    fprintf("Habitat LR greater than species LR: %.2f%%\n", ...
        habitatStillGreater);
end


%% 12. Creating the Negative Binomial Negative Log-Likelihood Function

function nLL = negativeBinomialNLL(parameters, X, Y)

    % Extracting the regression coefficients
    beta = parameters(1:end-1);

    % Extracting the transformed dispersion parameter
    logAlpha = parameters(end);

    % Calculating the linear predictor
    eta = X * beta;

    % Rejecting numerically unsafe parameter values
    if any(~isfinite(eta)) || any(abs(eta) > 500) || ...
            ~isfinite(logAlpha) || abs(logAlpha) > 30
        nLL = realmax;
        return;
    end

    % Calculating the expected parasite counts
    mu = exp(eta);

    % Calculating the dispersion and size parameters
    alpha = exp(logAlpha);
    r = 1 / alpha;

    % Calculating the negative binomial log-likelihood
    logLikelihood = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* (log(r) - log(r + mu)) ...
        + Y .* (eta - log(r + mu));

    % Summing and negating the log-likelihood
    nLL = -sum(logLikelihood);

    % Rejecting invalid likelihood values
    if ~isfinite(nLL)
        nLL = realmax;
    end
end