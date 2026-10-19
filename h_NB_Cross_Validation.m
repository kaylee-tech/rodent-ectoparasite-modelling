%% 1. Loading the prepared Data

% Loading the prepared rodent dataset
load("parasiteModel.mat", "parasiteModel");


%% 2. Extracting the required variables 

% Response variable
parasiteLoad = parasiteModel.Total_parasitical_load;

% Displaying the number of observations
fprintf("Number of observations: %d\n", height(modelData));
% Explanatory variables
species = categorical(parasiteModel.SPECIES);
habitats = categorical(parasiteModel.HBT_TYPE);
sex = categorical(parasiteModel.SEX);
breeding = categorical(parasiteModel.BRS);
totalLength = parasiteModel.TL;

% Creating the model data table
modelData = table( ...
    parasiteLoad, species, habitats, sex, breeding, totalLength);

% Extracting the response variable
responseVariable = modelData.parasiteLoad;

% Counting observations
numObservations = length(responseVariable);


%% 3. Constructing the Design Matrices

% Creating dummy variables
speciesDummy = dummyvar(modelData.species);
habitatDummy = dummyvar(modelData.habitats);
sexDummy = dummyvar(modelData.sex);
breedingDummy = dummyvar(modelData.breeding);

% Creating the intercept
intercept = ones(numObservations, 1);

% Constructing the full NB design matrix
Xfull = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    totalLength];

% Constructing the model without habitat
XnoHabitat = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    totalLength];

% Constructing the model without species
XnoSpecies = [ ...
    intercept, ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    totalLength];


%% 3. Creating Five Cross-Validation Folds

% Setting the random seed for reproducibility
rng(42);   % My Good luck number

% Creating five approximately equal folds
cv = cvpartition(numObservations, 'KFold', 5);

% Counting the number of folds
numFolds = cv.NumTestSets;

% Initialising matrices to store held-out NLL
% Each row represents a fold
% Each column represents a model
foldNLL = NaN(numFolds, 4);

% Initialising a vector for the number of test observations
foldTestSizes = zeros(numFolds, 1);

% Defining the model names
modelNames = [ ...
    "Full NB";
    "NB Without Habitat";
    "NB Without Species";
    "Habitat-Dependent Hurdle NB"];

% Setting optimisation options
options = optimset( ...
    'Display', 'off', ...
    'MaxFunEvals', 20000, ...
    'MaxIter', 20000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);


%% 4. Cross-Validating the Negative Binomial Models

% Storing the three design matrices
designMatrices = {Xfull, XnoHabitat, XnoSpecies};

% Repeating the analysis for each fold
for k = 1:numFolds

    % Identifying training and test observations
    trainIndex = training(cv, k);
    testIndex = test(cv, k);

    % Extracting training and test parasite counts
    Ytrain = Y(trainIndex);
    Ytest = Y(testIndex);

    % Recording the number of test observations
    foldTestSizes(k) = sum(testIndex);

    % Fitting each negative binomial model
    for j = 1:3

        % Selecting the current design matrix
        Xcurrent = designMatrices{j};

        % Extracting training and test predictors
        Xtrain = Xcurrent(trainIndex, :);
        Xtest = Xcurrent(testIndex, :);

        % Initialising regression coefficients
        initialBeta = Xtrain \ log(Ytrain + 0.5);

        % Initialising the dispersion parameter
        initialLogAlpha = log(1);

        % Combining initial parameters
        initialParameters = [initialBeta; initialLogAlpha];

        % Defining the training objective function
        objectiveFunction = @(parameters) ...
            negativeBinomialNLL(parameters, Xtrain, Ytrain);

        % Fitting the model using training observations
        [estimatedParameters, ~, exitflag] = ...
            fminsearch(objectiveFunction, initialParameters, options);

        % Checking optimisation success
        if exitflag <= 0
            warning("Model %d did not converge in fold %d.", j, k);
            continue;
        end

        % Evaluating the fitted model on unseen test observations
        foldNLL(k, j) = ...
            negativeBinomialNLL(estimatedParameters, Xtest, Ytest);
    end
end


%% 5. Cross-Validating the Habitat-Dependent Hurdle NB Model

% Creating a fixed categorical habitat variable
habitatCategories = categorical(modelData.habitats);

% Obtaining the habitat names
habitatNames = categories(habitatCategories);

% Counting the number of habitats
numHabitats = numel(habitatNames);

% Repeating the analysis for each fold
for k = 1:numFolds

    % Identifying training and test observations
    trainIndex = training(cv, k);
    testIndex = test(cv, k);

    % Extracting training and test parasite counts
    Ytrain = Y(trainIndex);
    Ytest = Y(testIndex);

    % Extracting the full design matrices
    Xtrain = Xfull(trainIndex, :);
    Xtest = Xfull(testIndex, :);

    % Extracting habitat categories
    habitatTrain = habitatCategories(trainIndex);
    habitatTest = habitatCategories(testIndex);

    % Identifying positive parasite counts in training data
    positiveTrain = Ytrain > 0;

    % Extracting positive training observations
    YpositiveTrain = Ytrain(positiveTrain);
    XpositiveTrain = Xtrain(positiveTrain, :);

    % Initialising habitat-specific occurrence probabilities
    habitatProbabilities = zeros(numHabitats, 1);

    % Estimating occurrence probabilities using training data only
    for h = 1:numHabitats

        % Identifying training rodents in the current habitat
        currentHabitat = habitatTrain == habitatNames{h};

        % Counting training rodents in this habitat
        nHabitat = sum(currentHabitat);

        % Counting parasite-positive rodents
        nPositive = sum(Ytrain(currentHabitat) > 0);

        % Using the half-count correction
        habitatProbabilities(h) = ...
            (nPositive + 0.5) / (nHabitat + 1);
    end

    % Initialising predicted occurrence probabilities for test data
    qTest = zeros(length(Ytest), 1);

    % Assigning habitat-specific probabilities to test rodents
    for h = 1:numHabitats
        currentHabitat = habitatTest == habitatNames{h};
        qTest(currentHabitat) = habitatProbabilities(h);
    end

    % Initialising the truncated NB parameters
    initialBeta = XpositiveTrain \ log(YpositiveTrain + 0.5);
    initialParameters = [initialBeta; log(1)];

    % Defining the zero-truncated NB objective function
    objectiveFunction = @(parameters) ...
        truncatedNegativeBinomialNLL( ...
            parameters, XpositiveTrain, YpositiveTrain);

    % Fitting the positive-count component
    [estimatedParameters, ~, exitflag] = ...
        fminsearch(objectiveFunction, initialParameters, options);

    % Checking convergence
    if exitflag <= 0
        warning("Hurdle model did not converge in fold %d.", k);
        continue;
    end

    % Identifying zero and positive counts in test data
    zeroTest = Ytest == 0;
    positiveTest = Ytest > 0;

    % Calculating log-likelihood for zero observations
    logLikelihoodZeros = sum(log(1 - qTest(zeroTest)));

    % Calculating occurrence log-likelihood for positive observations
    logLikelihoodOccurrence = sum(log(qTest(positiveTest)));

    % Calculating the zero-truncated NB log-likelihood for positives
    logLikelihoodPositiveCounts = ...
        -truncatedNegativeBinomialNLL( ...
            estimatedParameters, ...
            Xtest(positiveTest, :), ...
            Ytest(positiveTest));

    % Combining the hurdle log-likelihood components
    totalTestLogLikelihood = ...
        logLikelihoodZeros + ...
        logLikelihoodOccurrence + ...
        logLikelihoodPositiveCounts;

    % Storing the held-out negative log-likelihood
    foldNLL(k, 4) = -totalTestLogLikelihood;
end


%% 6. Summarising Cross-Validation Results

% Calculating the total held-out negative log-likelihood
totalNLL = sum(foldNLL, 1, 'omitnan')';

% Calculating the average held-out NLL per observation
meanNLL = totalNLL / numObservations;

% Counting the number of successfully evaluated folds
successfulFolds = sum(isfinite(foldNLL), 1)';

% Creating the comparison table
crossValidationTable = table( ...
    modelNames, ...
    successfulFolds, ...
    totalNLL, ...
    meanNLL, ...
    'VariableNames', ...
    {'Model', 'SuccessfulFolds', ...
     'TotalTestNLL', 'MeanNLLPerObservation'});

% Displaying results
fprintf('\n');
disp("----------------------------------");
disp("Five-Fold Cross-Validation Results");
disp("----------------------------------");
disp(crossValidationTable);


%% 7. Zero-Truncated Negative Binomial Negative Log-Likelihood Function
function nLL = truncatedNegativeBinomialNLL(parameters, X, Y)

    % Extracting regression coefficients and dispersion
    beta = parameters(1:end-1);
    alpha = exp(parameters(end));

    % Calculating expected parasite counts
    eta = X * beta;
    mu = exp(eta);

    % Converting dispersion to NB size parameter
    r = 1 / alpha;

    % Calculating the NB log-probability for positive counts
    logNB = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* (log(r) - log(r + mu)) ...
        + Y .* (eta - log(r + mu));

    % Calculating the probability of zero
    logPZero = r .* (log(r) - log(r + mu));

    % Calculating the log-probability of a positive count
    logPPositive = log(-expm1(logPZero));

    % Calculating the zero-truncated NB log-likelihood
    logTruncatedNB = logNB - logPPositive;

    % Returning the negative log-likelihood
    nLL = -sum(logTruncatedNB);

    % Penalising invalid parameter combinations
    if ~isfinite(nLL)
        nLL = realmax;
    end
end


%% 8. Creating the Negative Binomial - Negative Log-Likelihood Function

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