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


%% 4. Initialising the Zero-Inflated Negative Binomial Parameters

% The initial estimate for the dispersion parameter
initialAlpha = 1;

% Making sure alpha remains positive
initialLogAlpha = log(initialAlpha);

% Initial estimate for the additional zero-inflation probability
initialPi = 0.05;

% Transforming the probability onto the logit scale
initialLogitPi = log(initialPi / (1 - initialPi));

% Combining all starting parameters
initialParameters = [initialBeta; initialLogAlpha; initialLogitPi];

% Displaying the number of parameters
fprintf("Number of ZINB parameters: %d\n", ...
    numel(initialParameters));


%% 5. Fitting the Zero-Inflated Negative Binomial Model

% Creating the negative log-likelihood objective function
objectiveFunction = @(parameters) zinbNLL(parameters, X, Y);

% Setting optimisation options
options = optimset('Display', 'off', ...
    'MaxFunEvals', 30000, ...
    'MaxIter', 30000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Selecting different initial zero-inflation probabilities
startingPiValues = [0.01, 0.05, 0.10, 0.20, 0.30];

% Initialising storage for optimisation results
allParameters = zeros(numel(initialParameters), ...
    numel(startingPiValues));

allNegativeLogLikelihoods = zeros(numel(startingPiValues), 1);
allExitFlags = zeros(numel(startingPiValues), 1);

% Fitting the model from different starting values
for i = 1:numel(startingPiValues)

    % Selecting the current initial zero-inflation probability
    initialPi = startingPiValues(i);

    % Converting the probability to the logit scale
    initialLogitPi = log(initialPi / (1 - initialPi));

    % Constructing the initial parameter vector
    currentInitialParameters = ...
        [initialBeta; initialLogAlpha; initialLogitPi];

    % Fitting the ZINB model
    [parameters, nLL, exitflagCurrent] = ...
        fminsearch(objectiveFunction, currentInitialParameters, options);

    % Storing the fitted parameters and likelihood
    allParameters(:, i) = parameters;
    allNegativeLogLikelihoods(i) = nLL;
    allExitFlags(i) = exitflagCurrent;

end

% Finding the fit with the smallest negative log-likelihood
[negativeLogLikelihood, bestIndex] = ...
    min(allNegativeLogLikelihoods);

% Extracting the best fitted parameters
estimatedParameters = allParameters(:, bestIndex);

% Extracting the fitted zero-inflation probability for each run
fittedPiValues = 1 ./ ...
    (1 + exp(-allParameters(end, :)'));

% Displaying the results with greater precision
resultsTable = table( ...
    startingPiValues(:), ...
    fittedPiValues, ...
    -allNegativeLogLikelihoods, ...
    allExitFlags, ...
    'VariableNames', ...
    {'StartingPi', 'FittedPi', 'LogLikelihood', 'ExitFlag'});

format long g
disp(resultsTable);


%% 6. Extracting the ZINB Parameter Estimates

% Extracting the regression coefficients
estimatedBeta = estimatedParameters(1:end-2);

% Extracting the dispersion parameter
estimatedAlpha = exp(estimatedParameters(end-1));

% Extracting the zero-inflation probability
estimatedPi = 1 / (1 + exp(-estimatedParameters(end)));

% Calculating the NB component's expected parasite counts
countMean = exp(X * estimatedBeta);

% Calculating the overall ZINB expected parasite counts
predictedCounts = (1 - estimatedPi) .* countMean;

% Displaying the regression coefficients
coefficientNames = poissonModel.CoefficientNames(:);

coefficientTable = table(coefficientNames, estimatedBeta, ...
    'VariableNames', {'Coefficient', 'Estimate'});

fprintf('\n');
disp("----------------------------------------------");
disp("Zero-Inflated Negative Binomial Model Results");
disp("----------------------------------------------");

disp(coefficientTable);

fprintf("Estimated dispersion (alpha): %.4f\n", estimatedAlpha);
fprintf("Estimated zero-inflation probability: %.4f\n", estimatedPi);
fprintf("Log-likelihood: %.2f\n", -negativeLogLikelihood);
fprintf("Optimisation exit flag: %d\n", allExitFlags(bestIndex));


%% 7. Creating the Zero-Inflated Negative Binomial - Negative Log-Likelihood Function

function nLL = zinbNLL(parameters, X, Y)

    % Extracting the regression coefficients
    beta = parameters(1:end-2);

    % Extracting the transformed dispersion parameter
    logAlpha = parameters(end-1);

    % Extracting the transformed zero-inflation parameter
    logitPi = parameters(end);

    % Calculating the linear predictor
    eta = X * beta;

    % Rejecting numerically unsafe parameter values
    if any(~isfinite(eta)) || any(abs(eta) > 500) || ...
            ~isfinite(logAlpha) || abs(logAlpha) > 30 || ...
            ~isfinite(logitPi) || abs(logitPi) > 30

        nLL = realmax;
        return;
    end

    % Calculating expected parasite counts
    mu = exp(eta);

    % Calculating the dispersion parameter
    alpha = exp(logAlpha);

    % Calculating the zero-inflation probability
    piZero = 1 / (1 + exp(-logitPi));

    % Converting dispersion to the NB size parameter
    r = 1 / alpha;

    % Calculating the negative binomial log-probabilities
    logNB = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* (log(r) - log(r + mu)) ...
        + Y .* (eta - log(r + mu));

    % Identifying zero and positive parasite counts
    zeroIndex = (Y == 0);
    positiveIndex = (Y > 0);

    % Initialising the ZINB log-likelihood contributions
    logLikelihood = zeros(size(Y));

    % Likelihood for zero parasite counts
    logLikelihood(zeroIndex) = log( ...
        piZero + (1 - piZero) .* exp(logNB(zeroIndex)));

    % Likelihood for positive parasite counts
    logLikelihood(positiveIndex) = ...
        log(1 - piZero) + logNB(positiveIndex);

    % Calculating the total negative log-likelihood
    nLL = -sum(logLikelihood);

    % Rejecting non-finite likelihood values
    if ~isfinite(nLL)
        nLL = realmax;
    end
end