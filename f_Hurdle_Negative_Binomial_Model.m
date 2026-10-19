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

% Checking the design matrix matches the Poisson model
poissonPredictions = predict(poissonModel, modelData);
matrixPredictions = exp(X * initialBeta);

maximumDifference = max(abs(poissonPredictions - matrixPredictions));

fprintf("Maximum prediction difference: %.8f\n", maximumDifference);

if maximumDifference > 1e-6
    error("Design matrix does not match the Poisson model.");
end

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


%% 4. Preparing the Hurdle Model Components

% Identifying rodents with zero and positive parasite counts
zeroIndex = (Y == 0);
positiveIndex = (Y > 0);

% Extracting the positive parasite counts
Ypositive = Y(positiveIndex);

% Extracting predictor information for positive observations
Xpositive = X(positiveIndex, :);

% Creating the binary parasite occurrence variable
parasitePresent = double(Y > 0);

% Displaying the sample sizes
fprintf("Total observations: %d\n", length(Y));
fprintf("Zero parasite counts: %d\n", sum(zeroIndex));
fprintf("Positive parasite counts: %d\n", sum(positiveIndex));


%% 5. Fitting the Habitat-Dependent Parasite Occurrence Component

% Creating a categorical habitat variable
habitatCategories = categorical(modelData.habitats);

% Obtaining the names of the habitat categories
habitatNames = categories(habitatCategories);

% Counting the number of habitat categories
numHabitats = numel(habitatNames);

% Initialising vectors to store habitat-specific results
habitatSampleSizes = zeros(numHabitats, 1);
habitatPositiveCounts = zeros(numHabitats, 1);
habitatZeroCounts = zeros(numHabitats, 1);
habitatProbabilities = zeros(numHabitats, 1);


%% 5.1 Calculating Parasite Occurrence for Each Habitat

% Calculating the parasite occurrence probability for each habitat
for i = 1:numHabitats

    % Identifying rodents belonging to the current habitat
    currentHabitat = habitatCategories == habitatNames{i};

    % Counting the number of rodents in the habitat
    habitatSampleSizes(i) = sum(currentHabitat);

    % Counting rodents with at least one parasite
    habitatPositiveCounts(i) = sum(parasitePresent(currentHabitat));

    % Counting rodents with zero parasites
    habitatZeroCounts(i) = habitatSampleSizes(i) ...
    - habitatPositiveCounts(i);

    % Calculating the probability of parasite presence
    habitatProbabilities(i) = ...
        habitatPositiveCounts(i) / habitatSampleSizes(i);
end

% Calculating the probability of parasite presence
habitatProbabilities(i) = ...
    habitatPositiveCounts(i) / habitatSampleSizes(i);


% Creating a table of habitat-specific occurrence results
habitatOccurrenceTable = table( ...
    habitatNames, ...
    habitatSampleSizes, ...
    habitatPositiveCounts, ...
    habitatZeroCounts, ...
    habitatProbabilities, ...
    'VariableNames', ...
    {'Habitat', 'SampleSize', 'PositiveCounts', ...
     'ZeroCounts', 'OccurrenceProbability'});

% Displaying the habitat occurrence results
fprintf('\n');
disp("----------------------------------------------");
disp("Habitat-Dependent Parasite Occurrence Results");
disp("----------------------------------------------");

disp(habitatOccurrenceTable);


%% 5.2 Calculating Habitat Occurrence Regression Coefficients

% Applying a half-count correction to obtain finite log-odds
adjustedHabitatProbabilities = ...
    (habitatPositiveCounts + 0.5) ./ ...
    (habitatSampleSizes + 1);

% Converting the adjusted probabilities into log-odds
habitatLogOdds = log( ...
    adjustedHabitatProbabilities ./ ...
    (1 - adjustedHabitatProbabilities));

% Using Agriculture as the reference habitat
occurrenceIntercept = habitatLogOdds(1);

% Calculating the coefficients relative to Agriculture
occurrenceCoefficients = habitatLogOdds(2:end) ...
    - occurrenceIntercept;

% Combining the intercept and habitat coefficients
estimatedGamma = ...
    [occurrenceIntercept; occurrenceCoefficients];

% Creating coefficient names
occurrenceCoefficientNames = ...
    ["(Intercept)"; string(habitatNames(2:end))];

% Creating a table of the occurrence regression coefficients
occurrenceCoefficientTable = table( ...
    occurrenceCoefficientNames, ...
    estimatedGamma, ...
    'VariableNames', {'Coefficient', 'Estimate'});

% Displaying the occurrence regression coefficients
fprintf('\n');
disp("----------------------------------------------");
disp("Habitat Occurrence Regression Coefficients");
disp("----------------------------------------------");

disp(occurrenceCoefficientTable);


%% 5.3 Calculating Predicted Parasite Occurrence Probabilities

% Initialising a vector to store predicted probabilities
predictedOccurrence = zeros(length(Y), 1);

% Assigning habitat-specific probabilities to each rodent
for i = 1:numHabitats

    % Identifying rodents belonging to the current habitat
    currentHabitat = habitatCategories == habitatNames{i};

    % Assigning the corresponding habitat occurrence probability
    predictedOccurrence(currentHabitat) = ...
    adjustedHabitatProbabilities(i);

end


%% 5.4 Calculating the Occurrence Log-Likelihood

% Calculating the Bernoulli log-likelihood for parasite occurrence
occurrenceLogLikelihood = sum( ...
    parasitePresent .* log(predictedOccurrence) + ...
    (1 - parasitePresent) .* log(1 - predictedOccurrence));

% Displaying the occurrence log-likelihood
fprintf("Habitat occurrence log-likelihood: %.4f\n", ...
    occurrenceLogLikelihood);


%% 6. Initialising the Zero-Truncated Negative Binomial Parameters

% Using the Poisson regression coefficients as starting values
initialBetaTruncated = initialBeta;

% Initial estimate for dispersion
initialAlphaTruncated = 1;

% Transforming alpha to keep it positive
initialLogAlphaTruncated = log(initialAlphaTruncated);

% Combining the starting parameters
initialParametersTruncated = ...
    [initialBetaTruncated; initialLogAlphaTruncated];

% Displaying the number of parameters
fprintf("Number of truncated NB parameters: %d\n", ...
    numel(initialParametersTruncated));


%% 7. Fitting the Zero-Truncated Negative Binomial Model

% Creating the negative log-likelihood function
objectiveFunction = @(parameters) ...
    truncatedNegativeBinomialNLL(parameters, Xpositive, Ypositive);

% Setting optimisation options
options = optimset('Display', 'iter', ...
    'MaxFunEvals', 30000, ...
    'MaxIter', 30000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Estimating the zero-truncated NB parameters
[estimatedParametersTruncated, negativeLogLikelihoodTruncated, ...
    exitflagTruncated] = ...
    fminsearch(objectiveFunction, initialParametersTruncated, options);

% Checking whether optimisation actually converged
if exitflagTruncated ~= 1 || ~isfinite(negativeLogLikelihoodTruncated)
    error("Zero-truncated NB optimisation did not converge successfully.");
end


%% 8. Extracting the Hurdle Model Parameter Estimates

% Extracting the positive-count regression coefficients
estimatedBetaTruncated = estimatedParametersTruncated(1:end-1);

% Extracting the dispersion parameter
estimatedAlphaTruncated = exp(estimatedParametersTruncated(end));

% Creating the coefficient estimates table
coefficientNames = poissonModel.CoefficientNames(:);

coefficientTable = table(coefficientNames, ...
    estimatedBetaTruncated, ...
    'VariableNames', {'Coefficient', 'Estimate'});


%% 8.1 Calculating the Total Hurdle Model Log-Likelihood

% Combining the habitat occurrence and truncated NB log-likelihoods
totalLogLikelihood = occurrenceLogLikelihood ...
    - negativeLogLikelihoodTruncated;


%% 8.2 Calculating Model Selection Criteria

% Counting the parameters in the occurrence component
numOccurrenceParameters = numel(estimatedGamma);

% Counting the parameters in the positive-count component
numCountParameters = numel(estimatedParametersTruncated);

% Calculating the total number of estimated parameters
numParameters = numOccurrenceParameters + numCountParameters;

% Calculating the Akaike Information Criterion
AIC = 2 * numParameters - 2 * totalLogLikelihood;

% Calculating the Bayesian Information Criterion
BIC = log(length(Y)) * numParameters ...
    - 2 * totalLogLikelihood;


%% 8.3 Displaying the Hurdle Model Results
fprintf('\n');
disp("------------------------------------------------");
disp("Habitat-Dependent Hurdle Negative Binomial Model");
disp("------------------------------------------------");

% Displaying the occurrence component coefficients
disp("Habitat Occurrence Component:");
disp(occurrenceCoefficientTable);

% Displaying the positive-count component coefficients
disp("Zero-Truncated Negative Binomial Component:");
disp(coefficientTable);

% Displaying the estimated dispersion parameter
fprintf("Dispersion parameter (alpha): %.4f\n", ...
    estimatedAlphaTruncated);

% Displaying the component log-likelihoods
fprintf("Occurrence log-likelihood: %.4f\n", ...
    occurrenceLogLikelihood);

fprintf("Truncated NB log-likelihood: %.4f\n", ...
    -negativeLogLikelihoodTruncated);

% Displaying the combined hurdle log-likelihood
fprintf("Total log-likelihood: %.4f\n", ...
    totalLogLikelihood);

% Displaying the number of estimated parameters
fprintf("Occurrence parameters: %d\n", ...
    numOccurrenceParameters);

fprintf("Positive-count parameters: %d\n", ...
    numCountParameters);

fprintf("Total model parameters: %d\n", ...
    numParameters);

% Displaying model selection criteria
fprintf("AIC: %.4f\n", AIC);
fprintf("BIC: %.4f\n", BIC);


%% 9. Zero-Truncated Negative Binomial Negative Log-Likelihood Function

function nLL = truncatedNegativeBinomialNLL(parameters, X, Y)

    % Extracting regression coefficients
    beta = parameters(1:end-1);

    % Extracting transformed dispersion
    logAlpha = parameters(end);

    % Calculating linear predictor
    eta = X * beta;

    % Rejecting numerically unsafe parameter values
    if any(~isfinite(eta)) || any(abs(eta) > 500) || ...
            ~isfinite(logAlpha) || abs(logAlpha) > 30

        nLL = realmax;
        return;
    end

    % Calculating expected parasite counts
    mu = exp(eta);

    % Calculating dispersion and size parameter
    alpha = exp(logAlpha);
    r = 1 / alpha;

    % Calculating the ordinary NB log-probabilities
    logNB = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* (log(r) - log(r + mu)) ...
        + Y .* (eta - log(r + mu));

    % Calculating the NB probability of zero parasites
    logPZero = r .* (log(r) - log(r + mu));

    % Calculating log probability of a positive count
    logPPositive = log(-expm1(logPZero));

    % Calculating zero-truncated NB log-likelihood
    logLikelihood = logNB - logPPositive;

    % Summing and negating the log-likelihood
    nLL = -sum(logLikelihood);

    % Rejecting invalid likelihood values
    if ~isfinite(nLL)
        nLL = realmax;
    end

end