% 1. Loading and Preparing the Dataset

clear;
clc;

% Loading the original dataset
load("parasiteModel.mat", "parasiteModel");


%% 2. Extracting the required variables

% Extracting locality information
locality = categorical(parasiteModel.LOCALITY);

% Excluding My, which contains only Natural habitat
includeRows = locality ~= 'My';

% Extracting the response and explanatory variables
parasiteLoad = parasiteModel.Total_parasitical_load(includeRows);
species = categorical(parasiteModel.SPECIES(includeRows));
habitats = categorical(parasiteModel.HBT_TYPE(includeRows));
sex = categorical(parasiteModel.SEX(includeRows));
breeding = categorical(parasiteModel.BRS(includeRows));
totalLength = parasiteModel.TL(includeRows);
locality = removecats(locality(includeRows));

% Removing unused categorical levels
species = removecats(species);
habitats = removecats(habitats);
sex = removecats(sex);
breeding = removecats(breeding);

% Creating the model data table
modelData = table( ...
    parasiteLoad, species, habitats, locality, ...
    sex, breeding, totalLength);

% Displaying the number of observations
fprintf("Number of observations: %d\n", height(modelData));

% Checking habitat and locality distribution
disp(groupsummary(modelData, {'locality', 'habitats'}));


%% 3. Constructing Design Matrices

% Creating dummy variables for categorical predictors
speciesDummy = dummyvar(modelData.species);
habitatDummy = dummyvar(modelData.habitats);
localityDummy = dummyvar(modelData.locality);
sexDummy = dummyvar(modelData.sex);
breedingDummy = dummyvar(modelData.breeding);

% Creating the intercept
intercept = ones(height(modelData), 1);

% Extracting the response
Y = modelData.parasiteLoad;

% Full model: all explanatory variables
Xfull = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    localityDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Reduced model: without habitat
XnoHabitat = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    localityDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Reduced model: without species
XnoSpecies = [ ...
    intercept, ...
    habitatDummy(:, 2:end), ...
    localityDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Checking the ranks of the design matrices
fprintf("Full matrix: rank %d of %d columns\n", ...
    rank(Xfull), size(Xfull, 2));
fprintf("Without habitat: rank %d of %d columns\n", ...
    rank(XnoHabitat), size(XnoHabitat, 2));
fprintf("Without species: rank %d of %d columns\n", ...
    rank(XnoSpecies), size(XnoSpecies, 2));


%% 4. Fitting the Negative Binomial Models

% Storing the design matrices
designMatrices = {Xfull, XnoHabitat, XnoSpecies};

% Defining model names
modelNames = [ ...
    "Full NB + Locality";
    "NB Without Habitat";
    "NB Without Species"];

% Initialising result vectors
logLikelihood = NaN(3, 1);
AIC = NaN(3, 1);
BIC = NaN(3, 1);
numParameters = zeros(3, 1);
exitFlags = zeros(3, 1);

% Setting optimisation options
options = optimset( ...
    'Display', 'off', ...
    'MaxFunEvals', 30000, ...
    'MaxIter', 30000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Fitting each model
for j = 1:3

    % Selecting the current design matrix
    X = designMatrices{j};

    % Counting parameters, including dispersion
    numParameters(j) = size(X, 2) + 1;

    % Initialising the regression coefficients
    initialBeta = X \ log(Y + 0.5);

    % Initialising the dispersion parameter
    initialParameters = [initialBeta; log(1)];

    % Defining the negative log-likelihood function
    objectiveFunction = @(parameters) ...
        negativeBinomialNLL(parameters, X, Y);

    % Optimising model parameters
    [estimatedParameters, minimumNLL, exitflag] = ...
        fminsearch(objectiveFunction, initialParameters, options);

    % Storing convergence information
    exitFlags(j) = exitflag;

    % Storing results only for successful optimisations
    if exitflag > 0 && isfinite(minimumNLL)
        logLikelihood(j) = -minimumNLL;
        AIC(j) = 2*numParameters(j) - 2*logLikelihood(j);
        BIC(j) = log(length(Y))*numParameters(j) ...
            - 2*logLikelihood(j);
    else
        warning("Model %d did not converge successfully.", j);
    end
end

% Creating the model comparison table
resultsTable = table( ...
    modelNames, numParameters, logLikelihood, ...
    AIC, BIC, exitFlags);
fprintf('\n');
disp("-------------------------------------");
disp("Locality-Adjusted NB Model Comparison");
disp("-------------------------------------");
disp(resultsTable);


%% 5. Habitat Versus Species: Likelihood-Ratio Tests

% Proceed only if all three models converged
if all(exitFlags > 0) && all(isfinite(logLikelihood))

    % Calculating likelihood-ratio statistics
    LRhabitat = 2*(logLikelihood(1) - logLikelihood(2));
    LRspecies = 2*(logLikelihood(1) - logLikelihood(3));

    % Calculating degrees of freedom from parameter counts
    dfHabitat = numParameters(1) - numParameters(2);
    dfSpecies = numParameters(1) - numParameters(3);

    % Calculating p-values using the chi-square distribution
    pHabitat = gammainc(LRhabitat/2, dfHabitat/2, 'upper');
    pSpecies = gammainc(LRspecies/2, dfSpecies/2, 'upper');

    % Calculating the difference in contributions
    differenceLR = LRhabitat - LRspecies;

    % Displaying results
    fprintf('\n');
    disp("-------------------------------------");
    disp("Locality-Adjusted Variable Importance");
    disp("-------------------------------------");
    fprintf("Habitat LR: %.4f (df = %d, p = %.6g)\n", ...
        LRhabitat, dfHabitat, pHabitat);
    fprintf("Species LR: %.4f (df = %d, p = %.6g)\n", ...
        LRspecies, dfSpecies, pSpecies);
    fprintf("Difference (Habitat - Species): %.4f\n", ...
        differenceLR);
else
    warning("Likelihood-ratio tests skipped due to fitting problems.");
end


%% 6. Checking Species-Habitat Overlap

% Counting observations for each species-habitat combination
speciesHabitatCounts = groupsummary( ...
    modelData, {'species', 'habitats'});
fprintf('\n');
disp("----------------------------");
disp("Species-Habitat Distribution");
disp("----------------------------");
disp(speciesHabitatCounts);


%% 7. Matched-Sample Comparison Without Locality
%{ 
The same 298 rodents are used, but locality is excluded.
This allows us to isolate the effect of adding locality.
%}

% Constructing design matrices without locality
% Construct the reduced design matrix excluding habitat and locality
XfullNoLocality = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Construct the reduced design matrix excluding habitat and locality
XnoHabitatNoLocality = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Construct the reduced design matrix excluding species and locality
XnoSpeciesNoLocality = [ ...
    intercept, ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Storing the design matrices
matricesNoLocality = { ...
    XfullNoLocality, ...
    XnoHabitatNoLocality, ...
    XnoSpeciesNoLocality};

% Initialising results
logLikNoLocality = NaN(3, 1);
AICNoLocality = NaN(3, 1);
BICNoLocality = NaN(3, 1);
exitNoLocality = zeros(3, 1);

% Fitting the three models
for j = 1:3
    X = matricesNoLocality{j};

    % Checking identifiability
    if rank(X) < size(X, 2)
        error("Model %d has a rank-deficient design matrix.", j);
    end

    % Number of parameters, including NB dispersion
    k = size(X, 2) + 1;

    % Initial parameter estimates
    initialBeta = X \ log(Y + 0.5);
    initialParameters = [initialBeta; log(1)];

    % Defining the objective function
    objectiveFunction = @(parameters) ...
        negativeBinomialNLL(parameters, X, Y);

    % Fitting the negative binomial model
    [parameters, nLL, exitflag] = ...
        fminsearch(objectiveFunction, initialParameters, options);
    exitNoLocality(j) = exitflag;
    if exitflag > 0 && isfinite(nLL)
        logLikNoLocality(j) = -nLL;
        AICNoLocality(j) = 2*k + 2*nLL;
        BICNoLocality(j) = log(length(Y))*k + 2*nLL;
    else
        warning("Model %d did not converge.", j);
    end
end

% Displaying results
fprintf('\n');
disp("-----------------------------------------");
disp("Matched-Sample NB Models Without Locality");
disp("-----------------------------------------");
disp(table( ...
    modelNames, logLikNoLocality, ...
    AICNoLocality, BICNoLocality, exitNoLocality));


%% 8. Comparing Variable Importance With and Without Locality

if all(exitNoLocality > 0) && ...
        all(isfinite(logLikNoLocality)) && ...
        all(isfinite(logLikelihood))

    % Without locality
    LRhabitatNoLocality = ...
        2*(logLikNoLocality(1) - logLikNoLocality(2));
    LRspeciesNoLocality = ...
        2*(logLikNoLocality(1) - logLikNoLocality(3));

    % With locality (results from Section 4)
    LRhabitatWithLocality = LRhabitat;
    LRspeciesWithLocality = LRspecies;

    % Constructing the comparison table
    comparisonTable = table( ...
        ["Without Locality"; "With Locality"], ...
        [LRhabitatNoLocality; LRhabitatWithLocality], ...
        [LRspeciesNoLocality; LRspeciesWithLocality], ...
        'VariableNames', ...
        {'Model', 'HabitatLR', 'SpeciesLR'});
    fprintf('\n');
    disp("--------------------------------------------");
    disp("Matched-Sample Habitat vs Species Comparison");
    disp("--------------------------------------------");
    disp(comparisonTable);

    % Testing whether locality improves the full model
    LRlocality = 2*(logLikelihood(1) - logLikNoLocality(1));
    dfLocality = ...
        size(Xfull, 2) - size(XfullNoLocality, 2);
    pLocality = gammainc( ...
        max(LRlocality, 0)/2, dfLocality/2, 'upper');
    fprintf("\nLocality LR: %.4f\n", LRlocality);
    fprintf("Locality degrees of freedom: %d\n", dfLocality);
    fprintf("Locality p-value: %.6g\n", pLocality);
else
    warning("Comparison skipped because a model did not converge.");
end


%% 9. Creating the Negative Binomial - Negative Log-Likelihood Function

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