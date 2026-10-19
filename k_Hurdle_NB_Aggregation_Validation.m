%% 1. Load the simulation results

% Clearing the workspace and Command Window
clear;
clc;
close all;

% Defining the saved simulation filename
fileName = "Hurdle_NB_Simulation_Results.mat";

% Checking whether the file exists
if ~isfile(fileName)
    error("Simulation results file not found: %s", fileName);
end

% Loading the variables required for aggregation validation
load(fileName, ...
    "Y", "habitats", "simulatedCounts");

% Ensuring vectors have the correct shape
Y = Y(:);
habitats = categorical(habitats(:));

% Each rodent must have one habitat label
if numel(habitats) ~= numel(Y) || any(ismissing(habitats))
    error("Habitat labels are missing or do not match the observed counts.");
end

% Determining sample size and number of simulations
n = numel(Y);
numSimulations = size(simulatedCounts, 2);

% Checking simulation dimensions
if size(simulatedCounts, 1) ~= n || numSimulations == 0
    error("Observed and simulated sample sizes do not match.");
end

% Checking observed and simulated counts
if any(~isfinite(Y) | Y < 0 | Y ~= floor(Y)) || ...
        any(~isfinite(simulatedCounts(:)) | ...
        simulatedCounts(:) < 0 | ...
        simulatedCounts(:) ~= floor(simulatedCounts(:)))

    error("All parasite counts must be finite non-negative integers.");
end

% Identifying habitat categories
habitatNames = categories(habitats);
numHabitats = numel(habitatNames);

fprintf("----------------------------------------------\n");
fprintf("HURDLE NB AGGREGATION VALIDATION\n");
fprintf("----------------------------------------------\n");

fprintf("Number of rodents: %d\n", n);
fprintf("Number of simulated datasets: %d\n", numSimulations);


%% 2. Calculate aggregation in the observed data

% Calculating observed aggregation statistics
[observedTop10, observedTop20, observedGini] = ...
    calculateAggregation(Y);

% Storing observed statistics
observedOverall = [ ...
    observedTop10, observedTop20, observedGini];


%% 3. Calculate aggregation in the simulated data

% Initialising the results matrix
% Columns:
% 1 = Top 10% parasite share
% 2 = Top 20% parasite share
% 3 = Gini coefficient

simulatedOverall = NaN(numSimulations, 3);

% Calculating aggregation for each simulated dataset
for s = 1:numSimulations

    % Extracting parasite counts from simulation s
    Ysim = simulatedCounts(:, s);

    % Calculating aggregation statistics
    [top10, top20, gini] = calculateAggregation(Ysim);

    % Storing the results
    simulatedOverall(s, :) = [top10, top20, gini];

end

% Calculating simulated averages
overallSimulatedMean = mean(simulatedOverall, 1, "omitnan")';

% Calculating 95% simulation intervals
overallIntervals = prctile( ...
    simulatedOverall, [2.5 97.5], 1);

% Defining statistic names
overallStatisticNames = [ ...
    "Top 10% Parasite Share (%)";
    "Top 20% Parasite Share (%)";
    "Gini Coefficient"];

% Creating results table
overallResults = table( ...
    overallStatisticNames, ...
    observedOverall(:), ...
    overallSimulatedMean, ...
    overallIntervals(1, :)', ...
    overallIntervals(2, :)', ...
    'VariableNames', ...
    {'Statistic', 'Observed', 'SimulatedMean', ...
     'Lower95', 'Upper95'});

fprintf("\n----------------------------------------------\n");
fprintf("OVERALL AGGREGATION VALIDATION\n");
fprintf("----------------------------------------------\n");

disp(overallResults);


%% 4. Calculate aggregation for each habitat

% Initialising observed and simulated result matrices
observedHabitat = NaN(numHabitats, 3);

simulatedHabitat = NaN( ...
    numHabitats, numSimulations, 3);

% Looping through habitats
for h = 1:numHabitats

    % Identifying rodents in the habitat
    currentHabitat = habitats == habitatNames{h};

    % Extracting observed counts
    observedCounts = Y(currentHabitat);

    % Calculating observed aggregation
    [top10, top20, gini] = ...
        calculateAggregation(observedCounts);

    observedHabitat(h, :) = [top10, top20, gini];

    % Calculating aggregation in each simulation
    for s = 1:numSimulations

        % Extracting simulated counts in the habitat
        simulatedHabitatCounts = ...
            simulatedCounts(currentHabitat, s);

        % Calculating simulated aggregation
        [top10Sim, top20Sim, giniSim] = ...
            calculateAggregation(simulatedHabitatCounts);

        % Storing results
        simulatedHabitat(h, s, :) = ...
            [top10Sim, top20Sim, giniSim];

    end

end


%% 5. Summarise the habitat results

% Initialising summary arrays
habitatObservedTop20 = NaN(numHabitats, 1);
habitatSimulatedTop20 = NaN(numHabitats, 1);
habitatTop20Lower = NaN(numHabitats, 1);
habitatTop20Upper = NaN(numHabitats, 1);

habitatObservedGini = NaN(numHabitats, 1);
habitatSimulatedGini = NaN(numHabitats, 1);
habitatGiniLower = NaN(numHabitats, 1);
habitatGiniUpper = NaN(numHabitats, 1);

% Calculating summaries for each habitat
for h = 1:numHabitats

    % Extracting simulated top 20% shares
    top20Values = reshape( ...
        simulatedHabitat(h, :, 2), [], 1);

    % Extracting simulated Gini coefficients
    giniValues = reshape( ...
        simulatedHabitat(h, :, 3), [], 1);

    % Observed top 20% share
    habitatObservedTop20(h) = observedHabitat(h, 2);

    % Simulated top 20% statistics
    habitatSimulatedTop20(h) = mean(top20Values, "omitnan");
    top20Interval = prctile(top20Values, [2.5 97.5]);

    habitatTop20Lower(h) = top20Interval(1);
    habitatTop20Upper(h) = top20Interval(2);

    % Observed Gini
    habitatObservedGini(h) = observedHabitat(h, 3);

    % Simulated Gini statistics
    habitatSimulatedGini(h) = mean(giniValues, "omitnan");
    giniInterval = prctile(giniValues, [2.5 97.5]);

    habitatGiniLower(h) = giniInterval(1);
    habitatGiniUpper(h) = giniInterval(2);

end

% Creating summary table
habitatResults = table( ...
    string(habitatNames), ...
    habitatObservedTop20, ...
    habitatSimulatedTop20, ...
    habitatTop20Lower, ...
    habitatTop20Upper, ...
    habitatObservedGini, ...
    habitatSimulatedGini, ...
    habitatGiniLower, ...
    habitatGiniUpper, ...
    'VariableNames', ...
    {'Habitat', ...
     'ObservedTop20', 'SimulatedTop20', ...
     'Top20Lower95', 'Top20Upper95', ...
     'ObservedGini', 'SimulatedGini', ...
     'GiniLower95', 'GiniUpper95'});

fprintf("\n----------------------------------------------\n");
fprintf("HABITAT-SPECIFIC AGGREGATION VALIDATION\n");
fprintf("----------------------------------------------\n");

disp(habitatResults);


%% 6. Plot the overall top 20% parasite share

figure;

% Plotting simulated top 20% shares
histogram(simulatedOverall(:, 2), 25);

hold on;

% Adding observed top 20% share
xline(observedTop20, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed');

xlabel("Parasite Share Carried by Top 20% (%)");
ylabel("Frequency");

title("Hurdle NB: Top 20% Aggregation Validation");

grid on;
hold off;


%% 7. Plot the overall Gini coefficient

figure;

% Plotting simulated Gini coefficients
histogram(simulatedOverall(:, 3), 25);

hold on;

% Adding observed Gini coefficient
xline(observedGini, 'r', ...
    'LineWidth', 2, ...
    'Label', 'Observed');

xlabel("Gini Coefficient");
ylabel("Frequency");

title("Hurdle NB: Gini Coefficient Validation");

grid on;
hold off;


%% 8. Compare Gini coefficients across habitats

figure;

% Extracting the x-axis positions
x = 1:numHabitats;

% Plotting simulated mean Gini coefficients
bar(x, habitatSimulatedGini);

hold on;

% Calculating asymmetric error bars
lowerError = habitatSimulatedGini - habitatGiniLower;
upperError = habitatGiniUpper - habitatSimulatedGini;

% Adding 95% simulation intervals
errorbar(x, habitatSimulatedGini, ...
    lowerError, upperError, ...
    'k.', 'LineWidth', 1.5);

% Plotting observed Gini coefficients
plot(x, habitatObservedGini, 'ro', ...
    'MarkerSize', 8, ...
    'LineWidth', 2);

% Labelling the x-axis
xticks(x);
xticklabels(habitatNames);

xlabel("Habitat");
ylabel("Gini Coefficient");

title("Observed vs Simulated Gini by Habitat");

legend("Simulated Mean", "95% Simulation Interval", ...
    "Observed", 'Location', 'best');

ylim([0 1]);
grid on;
hold off;


%% 9. Save the results

% Saving MATLAB results
save("Hurdle_NB_Aggregation_Validation.mat", ...
    "overallResults", ...
    "habitatResults", ...
    "simulatedOverall", ...
    "simulatedHabitat");

% Exporting summary tables
writetable(overallResults, ...
    "Overall_Aggregation_Validation.csv");

writetable(habitatResults, ...
    "Habitat_Aggregation_Validation.csv");

fprintf("\n----------------------------------------------\n");
fprintf("AGGREGATION VALIDATION COMPLETED\n");
fprintf("----------------------------------------------\n");

fprintf("Results saved successfully.\n");


%% ============================================================
%  LOCAL FUNCTION: CALCULATING PARASITE AGGREGATION
%
%  This function calculates:
%  1. Top 10% parasite share
%  2. Top 20% parasite share
%  3. Gini coefficient
%
%  The function is included at the end of the script.
%% ============================================================

function [top10Share, top20Share, gini] = calculateAggregation(counts)

    % Ensuring the counts are a column vector
    counts = counts(:);

    % Number of rodents
    n = numel(counts);

    % Total number of parasites
    totalParasites = sum(counts);

    % Aggregation is undefined when there are no hosts or parasites
    if n == 0 || totalParasites == 0

        top10Share = NaN;
        top20Share = NaN;
        gini = NaN;

        return;

    end

    % Sorting parasite counts in descending order
    descendingCounts = sort(counts, 'descend');

    % Determining top 10% and 20% group sizes (rounding up)
    nTop10 = ceil(0.10 * n);
    nTop20 = ceil(0.20 * n);

    % Calculating top 10% parasite share
    top10Share = ...
        100 * sum(descendingCounts(1:nTop10)) ...
        / totalParasites;

    % Calculating top 20% parasite share
    top20Share = ...
        100 * sum(descendingCounts(1:nTop20)) ...
        / totalParasites;

    % Sorting counts in ascending order
    ascendingCounts = sort(counts, 'ascend');

    % Creating observation indices
    index = (1:n)';

    % Calculating the Gini coefficient
    gini = ...
        (2 * sum(index .* ascendingCounts)) ...
        / (n * totalParasites) ...
        - (n + 1) / n;
end