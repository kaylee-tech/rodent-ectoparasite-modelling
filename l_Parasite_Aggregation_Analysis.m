%% 1. Loading and Preparing the Dataset

% Clearing the workspace, Command Window and open figures
clear;
clc;
close all;

% Loading the original dataset
load("parasiteModel.mat", "parasiteModel");


%% 2. Extracting the required variables

% Extracting the observed parasite counts
parasiteLoad = parasiteModel.Total_parasitical_load;

% Ensuring parasite counts are stored as a column vector
parasiteLoad = parasiteLoad(:);

% Extracting habitat classifications
habitats = categorical(parasiteModel.HBT_TYPE);
habitats = habitats(:);

% Checking that parasite counts are finite, non-negative integers
invalidParasiteLoad = ...
    ~isfinite(parasiteLoad) | ...
    parasiteLoad < 0 | ...
    parasiteLoad ~= floor(parasiteLoad);
if any(invalidParasiteLoad)
    error("Parasite counts must be finite non-negative integers.");
end

% Checking that all rodents have a habitat classification
if any(isundefined(habitats))
    error("Some rodents have missing habitat classifications.");
end

% Checking that the parasite counts and habitats correspond
if numel(parasiteLoad) ~= numel(habitats)
    error("The number of parasite counts and habitat labels must match.");
end

% Determining the number of rodents and the total parasite count
n = numel(parasiteLoad);
totalParasites = sum(parasiteLoad);

% Aggregation measures are undefined when no parasites are recorded
if n == 0 || totalParasites == 0
    error("Aggregation cannot be calculated for an empty or parasite-free dataset.");
end
fprintf('\n');
disp("-----------------------------");
disp("Parasite Aggregation Analysis");
disp("-----------------------------");
fprintf("Number of rodents: %d\n", n);
fprintf("Total parasite count: %.0f\n", totalParasites);


%% 3. Calculating the overall Parasite Aggregation

% Sorting parasite counts from highest to lowest
sortedCounts = sort(parasiteLoad, 'descend');

% Determining how many rodents belong to the top 10% and 20%
nTop10 = ceil(0.10 * n);
nTop20 = ceil(0.20 * n);

% Calculating the number of parasites carried by these groups
parasitesTop10 = sum(sortedCounts(1:nTop10));
parasitesTop20 = sum(sortedCounts(1:nTop20));

% Expressing these counts as percentages of all recorded parasites
top10Share = 100 * parasitesTop10 / totalParasites;
top20Share = 100 * parasitesTop20 / totalParasites;

% Calculating the mean and variance of parasite abundance
overallMean = mean(parasiteLoad);
overallVariance = var(parasiteLoad);

% Calculating the variance-to-mean ratio
overallVMR = overallVariance / overallMean;

% Sorting parasite counts from lowest to highest for the Gini calculation
ascendingCounts = sort(parasiteLoad, 'ascend');

% Creating the observation indices
index = (1:n)';

%{ 
Calculating the Gini coefficient
A value of 0 indicates equal burdens; higher values indicate
greater inequality in the distribution of parasite counts.
%}
overallGini = ...
    (2 * sum(index .* ascendingCounts)) ...
    / (n * totalParasites) ...
    - (n + 1) / n;

% Displaying the overall aggregation statistics
fprintf('\n');
disp("----------------------------");
disp("Overall Parasite aggregation");
disp("----------------------------");
fprintf("Top 10%% of rodents: %d rodents\n", nTop10);
fprintf("Top 10%% parasite share: %.2f%%\n", top10Share);
fprintf("Top 20%% of rodents: %d rodents\n", nTop20);
fprintf("Top 20%% parasite share: %.2f%%\n", top20Share);
fprintf("Gini coefficient: %.4f\n", overallGini);
fprintf("Variance-to-mean ratio: %.4f\n", overallVMR);


%% 4. Calculating Habitat-Specific Parasite Aggregation

% Identifying the habitat categories
habitatNames = categories(habitats);
numHabitats = numel(habitatNames);

% Initialising vectors for the habitat-specific results
habitatSampleSize = zeros(numHabitats, 1);
habitatMean = NaN(numHabitats, 1);
habitatTop10Share = NaN(numHabitats, 1);
habitatTop20Share = NaN(numHabitats, 1);
habitatGini = NaN(numHabitats, 1);
habitatVMR = NaN(numHabitats, 1);

% Calculating aggregation statistics separately for each habitat
for h = 1:numHabitats

    % Identifying rodents belonging to the current habitat
    currentHabitat = habitats == habitatNames{h};

    % Extracting the parasite counts for these rodents
    counts = parasiteLoad(currentHabitat);

    % Calculating the habitat sample size and total parasite count
    nHabitat = numel(counts);
    totalHabitatParasites = sum(counts);

    % Storing the habitat sample size
    habitatSampleSize(h) = nHabitat;

    % Skipping categories with no sampled rodents
    if nHabitat == 0
        continue;
    end

    % Calculating the mean parasite count for the habitat
    habitatMean(h) = mean(counts);

    % Aggregation is undefined when no parasites were recorded
    if totalHabitatParasites == 0
        continue;
    end

    % Sorting parasite counts from highest to lowest
    sortedHabitatCounts = sort(counts, 'descend');

    % Determining the sizes of the top 10% and 20% groups
    nHabitatTop10 = ceil(0.10 * nHabitat);
    nHabitatTop20 = ceil(0.20 * nHabitat);

    % Calculating the parasite share carried by the top 10%
    habitatTop10Share(h) = ...
        100 * sum(sortedHabitatCounts(1:nHabitatTop10)) ...
        / totalHabitatParasites;

    % Calculating the parasite share carried by the top 20%
    habitatTop20Share(h) = ...
        100 * sum(sortedHabitatCounts(1:nHabitatTop20)) ...
        / totalHabitatParasites;

    % Sorting counts in ascending order for the Gini calculation
    ascendingHabitatCounts = sort(counts, 'ascend');

    % Creating the observation indices for this habitat
    indexHabitat = (1:nHabitat)';

    % Calculating the habitat-specific Gini coefficient
    habitatGini(h) = ...
        (2 * sum(indexHabitat .* ascendingHabitatCounts)) ...
        / (nHabitat * totalHabitatParasites) ...
        - (nHabitat + 1) / nHabitat;

    % Calculating the variance-to-mean ratio
    % At least two rodents are required for the sample variance.
    if nHabitat > 1
        habitatVMR(h) = var(counts) / mean(counts);
    end
end

% Creating a table of habitat-specific aggregation statistics
habitatAggregationResults = table( ...
    string(habitatNames), ...
    habitatSampleSize, ...
    habitatMean, ...
    habitatTop10Share, ...
    habitatTop20Share, ...
    habitatGini, ...
    habitatVMR, ...
    'VariableNames', ...
    {'Habitat', 'SampleSize', 'MeanParasiteLoad', ...
     'Top10Share', 'Top20Share', 'Gini', 'VMR'});

% Displaying the habitat-specific results
fprintf('\n');
disp("-------------------------------------");
disp(" Habitat-Specific Aggregation Results");
disp("-------------------------------------");
disp(habitatAggregationResults);


%% 5. Visualising the Top 20% Parasite Shares

% Creating a bar chart of the top 20% parasite share by habitat
figure;
bar(categorical(habitatAggregationResults.Habitat), ...
    habitatAggregationResults.Top20Share);
xlabel("Habitat");
ylabel("Percentage of Parasites (%)");
title("Parasites Carried by the Top 20% of Rodents");

% Setting the percentage range and displaying grid lines
ylim([0 100]);
grid on;


%% 6. Visualising Habitat-Specific Gini Coefficients 

% Creating a bar chart of Gini coefficients by habitat
figure;
bar(categorical(habitatAggregationResults.Habitat), ...
    habitatAggregationResults.Gini);
xlabel("Habitat");
ylabel("Gini Coefficient");
title("Parasite Burden Inequality by Habitat");

% Setting the Gini coefficient range and displaying grid lines
ylim([0 1]);
grid on;


%% 7. Visualising the Cumulative Parasite Burden

% Sorting parasite counts from highest to lowest
descendingCounts = sort(parasiteLoad, 'descend');

% Calculating the cumulative number of parasites
cumulativeParasites = cumsum(descendingCounts);

% Converting cumulative parasite counts into percentages
cumulativeParasiteShare = ...
    100 * cumulativeParasites / totalParasites;

% Calculating the cumulative percentage of rodents
cumulativeRodentShare = 100 * (1:n)' / n;

% Adding the origin so that the curve begins at zero
cumulativeRodentShare = [0; cumulativeRodentShare];
cumulativeParasiteShare = [0; cumulativeParasiteShare];

% Plotting the cumulative share of parasites carried by rodents
figure;
plot(cumulativeRodentShare, cumulativeParasiteShare, ...
    'LineWidth', 2);
hold on;

% Adding a reference line for equal parasite burdens
plot([0 100], [0 100], '--', 'LineWidth', 1.5);
xlabel("Cumulative Percentage of Rodents");
ylabel("Cumulative Percentage of Parasites");
title("Cumulative Parasite Burden Across Rodents");

% Identifying the observed curve and the equal-distribution line
legend("Observed Aggregation", "Equal Distribution", ...
    'Location', 'southeast');
xlim([0 100]);
ylim([0 100]);
grid on;
hold off;


%% 8. Saving the Results

% Saving the overall and habitat-specific results in MATLAB format
save("Parasite_Aggregation_Results.mat", ...
    "top10Share", ...
    "top20Share", ...
    "overallGini", ...
    "overallVMR", ...
    "habitatAggregationResults");

% Exporting the habitat-specific results as a CSV file
writetable(habitatAggregationResults, ...
    "Habitat_Aggregation_Results.csv");

% Confirming that the analysis has finished
fprintf('\n');
disp("------------------------------");
disp("Aggregation Analysis Completed");
disp("------------------------------");
fprintf("Results saved successfully.\n");