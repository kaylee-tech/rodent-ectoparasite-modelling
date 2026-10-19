%% 1. Loading and Preparing the Dataset

clear;
clc;
close all;

% Setting the random seed for reproducibility
rng(42);

% Loading the original rodent dataset
load("parasiteModel.mat", "parasiteModel");

% Extracting the response and explanatory variables
parasiteLoad = parasiteModel.Total_parasitical_load;
species = categorical(parasiteModel.SPECIES);
habitats = categorical(parasiteModel.HBT_TYPE);
sex = categorical(parasiteModel.SEX);
breeding = categorical(parasiteModel.BRS);
totalLength = parasiteModel.TL;

% Creating a table containing the model variables
modelData = table( ...
    parasiteLoad, ...
    species, ...
    habitats, ...
    sex, ...
    breeding, ...
    totalLength);

% Extracting parasite counts
Y = modelData.parasiteLoad;

% Determining the number of rodents
n = length(Y);

% Checking that parasite counts are valid
if any(~isfinite(Y) | Y < 0 | Y ~= floor(Y))
    error("Parasite counts must be non-negative integers.");
end

fprintf("--------------------------------------");
fprintf("HABITAT-DEPENDENT HURDLE NB SIMULATION");
fprintf("--------------------------------------");

fprintf("Number of rodents: %d\n", n);
fprintf("Observed zeros: %d\n", sum(Y == 0));
fprintf("Observed mean: %.4f\n", mean(Y));


%% 2. Constructing the Design Matrix

% Converting categorical variables into dummy variables
speciesDummy = dummyvar(modelData.species);
habitatDummy = dummyvar(modelData.habitats);
sexDummy = dummyvar(modelData.sex);
breedingDummy = dummyvar(modelData.breeding);

% Creating the intercept
intercept = ones(n, 1);

% Constructing the full design matrix
% Excluding the first category of each predictor as the reference

Xfull = [ ...
    intercept, ...
    speciesDummy(:, 2:end), ...
    habitatDummy(:, 2:end), ...
    sexDummy(:, 2:end), ...
    breedingDummy(:, 2:end), ...
    modelData.totalLength];

% Identifying rodents with positive parasite counts
positiveIndex = Y > 0;

% Extracting positive observations
Ypositive = Y(positiveIndex);
Xpositive = Xfull(positiveIndex, :);

% Checking the rank of the positive-count design matrix
fprintf("\nDesign matrix rank: %d of %d columns\n", ...
    rank(Xpositive), size(Xpositive, 2));

if rank(Xpositive) < size(Xpositive, 2)
    error("Positive-count design matrix is rank deficient.");
end


%% 3. Fitting the Zero-Truncated Negative Binomial

% Initialising regression coefficients
initialBeta = Xpositive \ log(Ypositive + 0.5);

% Initialising the dispersion parameter
initialLogAlpha = log(1);

% Combining initial parameters
initialParameters = [initialBeta; initialLogAlpha];

% Defining optimisation settings
options = optimset( ...
    'Display', 'off', ...
    'MaxFunEvals', 30000, ...
    'MaxIter', 30000, ...
    'TolX', 1e-8, ...
    'TolFun', 1e-8);

% Defining the negative log-likelihood function
objectiveFunction = @(parameters) ...
    truncatedNegativeBinomialNLL( ...
        parameters, Xpositive, Ypositive);

% Estimating the model parameters
[estimatedParameters, minimumNLL, exitflag] = ...
    fminsearch(objectiveFunction, initialParameters, options);

% Checking convergence
if exitflag <= 0 || ~isfinite(minimumNLL)
    error("The zero-truncated NB model did not converge.");
end

% Extracting regression coefficients
beta = estimatedParameters(1:end-1);

% Extracting dispersion
alpha = exp(estimatedParameters(end));

% Calculating the underlying NB mean for each rodent
mu = exp(Xfull * beta);

fprintf("-------------------------------");
fprintf("ZERO-TRUNCATED NB MODEL RESULTS");
fprintf("-------------------------------");

fprintf("Model converged: Yes\n");
fprintf("Estimated alpha: %.6f\n", alpha);
fprintf("Positive-count NLL: %.6f\n", minimumNLL);


%% 4. Estimating Habitat-Dependent Occurrence

% Obtaining habitat categories
habitatNames = categories(habitats);

% Counting habitat categories
numHabitats = numel(habitatNames);

% Initialising habitat occurrence probabilities
habitatProbabilities = zeros(numHabitats, 1);

% Initialising individual occurrence probabilities
q = zeros(n, 1);

% Estimating parasite occurrence probabilities by habitat
for h = 1:numHabitats

    % Identifying rodents in the current habitat
    currentHabitat = habitats == habitatNames{h};

    % Number of rodents in the habitat
    nHabitat = sum(currentHabitat);

    % Number of rodents with positive parasite counts
    nPositive = sum(Y(currentHabitat) > 0);

    % Half-count correction to avoid probabilities of 0 or 1
    habitatProbabilities(h) = ...
        (nPositive + 0.5) / (nHabitat + 1);

    % Assigning occurrence probability to each rodent
    q(currentHabitat) = habitatProbabilities(h);

end

% Displaying occurrence probabilities
occurrenceTable = table( ...
    string(habitatNames), ...
    habitatProbabilities, ...
    'VariableNames', ...
    {'Habitat', 'OccurrenceProbability'});

fprintf("---------------------------------");
fprintf("HABITAT OCCURRENCE PROBABILITIES");
fprintf("--------------------------------");

disp(occurrenceTable);


%% 5. Simulating Parasite Loads

% Number of simulated datasets
numSimulations = 1000;

% Initialising the simulated parasite-count matrix
% Each row represents a rodent and each column a simulation
simulatedCounts = zeros(n, numSimulations);

% Converting dispersion to NB size parameter
r = 1 / alpha;

% Calculating NB probability parameter
p = r ./ (r + mu);

% Calculating the underlying probability of zero
pZero = p .^ r;

% Simulating each dataset
for s = 1:numSimulations

    % Simulating whether each rodent carries parasites
    infested = rand(n, 1) < q;

    % Initialising parasite counts
    Ysim = zeros(n, 1);

    % Counting infested rodents
    nInfested = sum(infested);

    if nInfested > 0

        % Generating random uniform probabilities
        u = rand(nInfested, 1);

        % Mapping probabilities onto the positive NB distribution
        adjustedU = pZero(infested) + ...
            (1 - pZero(infested)) .* u;

        % Avoiding the exact upper endpoint of the CDF
        adjustedU = min(adjustedU, 1 - eps);

        % Generating strictly positive NB counts
        Ysim(infested) = nbininv( ...
            adjustedU, r, p(infested));

    end

    % Storing simulated parasite counts
    simulatedCounts(:, s) = Ysim;

end

% Checking that the simulation produced valid counts
if any(~isfinite(simulatedCounts(:))) || ...
        any(simulatedCounts(:) < 0)

    error("Simulation produced invalid parasite counts.");
end

fprintf("---------------------");
fprintf("SIMULATION COMPLETED");
fprintf("--------------------");

fprintf("Number of simulations: %d\n", numSimulations);
fprintf("Rodents per simulation: %d\n", n);


%% 6. Calculating Observed and Simulated Statistics

% Calculating observed descriptive statistics
observedMean = mean(Y);
observedVariance = var(Y);
observedZeros = sum(Y == 0);
observedMaximum = max(Y);
observedVMR = observedVariance / observedMean;

% Calculating descriptive statistics for each simulation
simulatedMeans = mean(simulatedCounts, 1);
simulatedVariances = var(simulatedCounts, 0, 1);
simulatedZeros = sum(simulatedCounts == 0, 1);
simulatedMaximums = max(simulatedCounts, [], 1);
simulatedVMR = simulatedVariances ./ simulatedMeans;

% Combining simulated statistics
simulatedStatistics = [ ...
    simulatedMeans(:), ...
    simulatedVariances(:), ...
    simulatedZeros(:), ...
    simulatedMaximums(:), ...
    simulatedVMR(:)];

% Combining observed statistics
observedStatistics = [ ...
    observedMean; ...
    observedVariance; ...
    observedZeros; ...
    observedMaximum; ...
    observedVMR];

% Calculating average simulated statistics
simulationAverages = mean(simulatedStatistics, 1)';

% Calculating 95% simulation intervals
simulationIntervals = prctile( ...
    simulatedStatistics, [2.5 97.5], 1);

% Defining statistic names
statisticNames = [ ...
    "Mean Parasite Load";
    "Variance";
    "Number of Zeros";
    "Maximum Parasite Load";
    "Variance-to-Mean Ratio"];

% Creating the summary table
simulationResults = table( ...
    statisticNames, ...
    observedStatistics, ...
    simulationAverages, ...
    simulationIntervals(1, :)', ...
    simulationIntervals(2, :)', ...
    'VariableNames', ...
    {'Statistic', 'Observed', 'SimulatedMean', ...
     'Lower95', 'Upper95'});

fprintf("----------------------------------------");
fprintf("SIMULATION-BASED GOODNESS-OF-FIT RESULTS");
fprintf("----------------------------------------");

disp(simulationResults);


%% 7. Visualising the Number of Zero Counts

figure;

% Plotting the simulated zero-count distribution
histogram(simulatedZeros, 25);

hold on;

% Adding the observed number of zeros
xline(observedZeros, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed Zeros');

xlabel("Number of Parasite-Free Rodents");
ylabel("Frequency");

title("Hurdle NB: Simulated vs Observed Zero Counts");

grid on;
hold off;


%% 8. Visualising the Simulated Mean Parasite Load

figure;

% Plotting the simulated mean distribution
histogram(simulatedMeans, 25);

hold on;

% Adding the observed mean
xline(observedMean, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed Mean');

xlabel("Mean Parasite Load");
ylabel("Frequency");

title("Hurdle NB: Simulated vs Observed Mean");

grid on;
hold off;


%% 9. Visualising the Simulated Maximum Parasite Load

figure;

% Plotting simulated maximum parasite counts
histogram(simulatedMaximums, 25);

hold on;

% Adding the observed maximum
xline(observedMaximum, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed Maximum');

xlabel("Maximum Parasite Load");
ylabel("Frequency");

title("Hurdle NB: Simulated vs Observed Maximum");

grid on;
hold off;


%% 10. Habitat-Specific Simulation Diagnostics

fprintf("---------------------------------------");
fprintf("HABITAT-SPECIFIC SIMULATION DIAGNOSTICS");
fprintf("---------------------------------------");

for h = 1:numHabitats

    % Identifying rodents in the habitat
    currentHabitat = habitats == habitatNames{h};

    % Extracting observed parasite counts
    observedHabitatCounts = Y(currentHabitat);

    % Calculating observed statistics
    observedHabitatMean = mean(observedHabitatCounts);
    observedHabitatZeros = sum(observedHabitatCounts == 0);

    % Extracting simulated counts for the habitat
    habitatSimulations = simulatedCounts(currentHabitat, :);

    % Calculating simulated statistics
    simulatedHabitatMeans = mean(habitatSimulations, 1);
    simulatedHabitatZeros = sum(habitatSimulations == 0, 1);

    % Calculating simulation intervals for the mean
    meanInterval = prctile( ...
        simulatedHabitatMeans, [2.5 97.5]);

    % Calculating simulation intervals for zero counts
    zeroInterval = prctile( ...
        simulatedHabitatZeros, [2.5 97.5]);

    % Displaying habitat-specific results
    fprintf("\nHabitat: %s\n", habitatNames{h});

    fprintf("Observed mean: %.2f\n", observedHabitatMean);

    fprintf("Simulated mean: %.2f\n", ...
        mean(simulatedHabitatMeans));

    fprintf("Simulated mean 95%% interval: [%.2f, %.2f]\n", ...
        meanInterval(1), meanInterval(2));

    fprintf("Observed zeros: %d\n", observedHabitatZeros);

    fprintf("Simulated zeros: %.2f\n", ...
        mean(simulatedHabitatZeros));

    fprintf("Simulated zeros 95%% interval: [%.0f, %.0f]\n", ...
        zeroInterval(1), zeroInterval(2));

end


%% 11. Comparing Observed and Simulated Distributions
% Using the same bin edges for observed and simulated counts

% Defining common bin edges
binEdges = [0:10:300, 350:50:1000];

% Calculating observed histogram proportions
observedHistogram = histcounts(Y, binEdges, ...
    'Normalization', 'probability');

% Calculating simulated histogram proportions
simulatedHistogram = histcounts( ...
    simulatedCounts(:), binEdges, ...
    'Normalization', 'probability');

% Calculating bin centres
binCentres = (binEdges(1:end-1) + ...
    binEdges(2:end)) / 2;

% Plotting distributions
figure;

plot(binCentres, observedHistogram, ...
    '-o', 'LineWidth', 2);

hold on;

plot(binCentres, simulatedHistogram, ...
    '-s', 'LineWidth', 2);

xlabel("Parasite Load");
ylabel("Proportion of Observations");

title("Observed vs Simulated Parasite-Load Distribution");

legend("Observed", "Simulated");

xlim([0 300]);
grid on;
hold off;


%% 12. Saving Simulation Results

% Defining the output filename
fileName = "Hurdle_NB_Simulation_Results.mat";

% Saving the fitted parameters and simulated datasets
save(fileName, ...
    "estimatedParameters", ...
    "alpha", ...
    "mu", ...
    "q", ...
    "habitatProbabilities", ...
    "simulatedCounts", ...
    "simulationResults", ...
    "Y", ...
    "habitats");

% Checking whether the file was created
if isfile(fileName)

    fprintf("\n--------------------------");
    fprintf("RESULTS SAVED SUCCESSFULLY\n");
    fprintf("----------------------------");

    fprintf("File location: %s\n", ...
        fullfile(pwd, fileName));

else

    error("The simulation results file was not created.");

end


%% 13. Variance-To-Mean Ratio Diagnostic

figure;

histogram(simulatedVMR, 25);

hold on;

xline(observedVMR, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed VMR');

xlabel("Variance-to-Mean Ratio");
ylabel("Frequency");

title("Hurdle NB: Simulated vs Observed Dispersion");

grid on;
hold off;


%% 14. Creating the Zero-Truncated Negative Binomial Negative Log-Likelihood Function

function nLL = truncatedNegativeBinomialNLL(parameters, X, Y)

    % Extracting regression coefficients
    beta = parameters(1:end-1);

    % Extracting dispersion parameter
    alpha = exp(parameters(end));

    % Calculating the linear predictor
    eta = X * beta;

    % Calculating the underlying NB mean
    mu = exp(eta);

    % Converting dispersion to size parameter
    r = 1 / alpha;

    % Calculating the NB log-probability
    logNB = gammaln(Y + r) ...
        - gammaln(r) ...
        - gammaln(Y + 1) ...
        + r .* (log(r) - log(r + mu)) ...
        + Y .* (eta - log(r + mu));

    % Calculating the log-probability of zero
    logPZero = r .* (log(r) - log(r + mu));

    % Calculating log(1 - P(Y = 0))
    % Using expm1 to improve numerical stability
    logPPositive = log(-expm1(logPZero));

    % Calculating zero-truncated log-probabilities
    logTruncatedNB = logNB - logPPositive;

    % Calculating negative log-likelihood
    nLL = -sum(logTruncatedNB);

    % Penalising invalid parameter combinations
    if ~isfinite(nLL)
        nLL = realmax;
    end
end