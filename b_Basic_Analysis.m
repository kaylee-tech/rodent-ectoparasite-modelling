%% 1. Loading the prepared data

% Loading the prepared rodent dataset
load("parasiteModel.mat", "parasiteModel")


%% 2. Extracting all the variables here

% Response variable
parasiteLoad = parasiteModel.Total_parasitical_load; 

% Potential predictors 
totalLength = parasiteModel.TL;
habitats = parasiteModel.HBT_TYPE;
species = parasiteModel.SPECIES;
breeding = parasiteModel.BRS;
sex = parasiteModel.SEX;
locality = parasiteModel.LOCALITY;

% Identifying observations with zero parasite counts
parasiteModel.ZeroParasites = parasiteLoad == 0;


%% 3. Descriptive statistics of the parasite load

% Calculating standard statistical measurements
sampleSizeParasite = numel(parasiteLoad);
meanParasite = mean(parasiteLoad);
medianParasite = median(parasiteLoad);
minParasite = min(parasiteLoad);
maxParasite = max(parasiteLoad);
varianceParasite = var(parasiteLoad);
stdParasite = std(parasiteLoad);
zeroParasite = sum(parasiteLoad == 0);
zeroPercentageParasite = 100 * zeroParasite / sampleSizeParasite;

% Calculating variance-to-mean (vtm) ratio
vtmParasite = varianceParasite / meanParasite;

% Displaying all the calculated results
disp("----------------------------------------------")
disp("Statistical Measurements on the Parasite Load:")
disp("----------------------------------------------")
fprintf("Parasite sample size: %d\n", sampleSizeParasite);
fprintf("Mean parasite count: %.2f\n", meanParasite);
fprintf("Median parasite count: %.2f\n", medianParasite);
fprintf("Minimum parasite count: %.2f\n", minParasite);
fprintf("Maximum parasite count: %.2f\n", maxParasite);
fprintf("Variance: %.2f\n", varianceParasite);
fprintf("Standard deviation: %.2f\n", stdParasite);
fprintf("Variance-to-mean ratio: %.2f\n", vtmParasite);
fprintf("Number of zeros: %d\n", zeroParasite);
fprintf("Percentage of zeros: %.2f%%\n", zeroPercentageParasite);



%% 4. Distribution of the Parasite Load

% Creating a histogram to examine shape 
figure;
histogram(parasiteLoad, "Binwidth", 5, ...
    "FaceColor", [0.85 0.15 0.15]);
xlabel("Ectoparasite count per rodent");
ylabel("Number of rodents");
title("Distribution of Ectoparasite Load");
grid on;
xlim([0 300]);

% Creating a boxplot 
figure;
boxplot(parasiteLoad, ...
    'Symbol', 'o', ...
    'OutlierSize', 3);
ylabel('Ectoparasite count per rodent');
title('Distribution of Ectoparasite Load');
grid on;

% Investigating the zero frequency in rodents

positiveCount = sum(parasiteLoad > 0);
figure;
bar([zeroParasite, positiveCount]);
set(gca, 'XTickLabel', {'Zero', 'Positive'});
ylabel('Number of rodents');
title('Frequency of zero and positive parasite counts');

% Looking at the Skewness and kurtosis
skewParasite = skewness(parasiteLoad);
kurtParasite = kurtosis(parasiteLoad);

% Displaying the results
fprintf('\n');
disp("------------------------------------------")
disp("Skewness and Kurtosis of the parasite load")
disp("------------------------------------------")
fprintf("Skewness: %.2f\n", skewParasite);
fprintf("Kurtosis: %.2f\n", kurtParasite);


% Calculating the Percentiles 
quantilesParasite = prctile(parasiteLoad, [25 50 75 90 95 99]);
percentileTable = table([25 50 75 90 95 99]', quantilesParasite', ...
    'VariableNames', {'Percentile', 'Parasite_Load'});

% Displaying the table
fprintf('\n');
disp("-----------------------------------------")
disp("Selected Percentiles of Ectoparasite Load");
disp("-----------------------------------------")
disp(percentileTable);

% Calculating the Interquartile Range
iqrParasite = iqr(parasiteLoad);
fprintf("Interquartile range: %.2f\n", iqrParasite);


%% 5. Comparing Ectoparasites Across Species

% Calculating descriptive statistics for each species
speciesStats = groupsummary(parasiteModel, 'SPECIES', ...
    {'mean', 'median', 'std', 'min', 'max'}, ...
    'Total_parasitical_load');

% Calculating zero counts and proportions
speciesZeros = groupsummary(parasiteModel, 'SPECIES', ...
    {'sum', 'mean'}, 'ZeroParasites');

% Converting zero proportions to percentages
speciesZeros.mean_ZeroParasites = ...
    speciesZeros.mean_ZeroParasites * 100;

% Joining tables
speciesSummary = join(speciesStats, ...
    speciesZeros(:, {'SPECIES', 'sum_ZeroParasites', ...
    'mean_ZeroParasites'}), ...
    'Keys', 'SPECIES');

% Displaying the results
fprintf('\n');
disp("----------------------------------------")
disp("Descriptive Statistics by Rodent Species")
disp("----------------------------------------")
disp(speciesSummary);


% Creating a boxplot comparing parasite load across species
% Colours for the four rodent species
colorsSpecies = [
    0.85 0.20 0.20;  % G. leucogaster - Red
    0.20 0.50 0.85;  % M. coucha - Blue
    0.30 0.65 0.35;  % M. natalensis - Green
    0.70 0.40 0.80   % S. campestris - Purple
];

figure;
boxplot(parasiteLoad, species, ...
    'Symbol', 'o', ...
    'OutlierSize', 3);

% Find the box objects
boxes = findobj(gca, 'Tag', 'Box');

% Colour each box
for k = 1:length(boxes)
    patch(get(boxes(k), 'XData'), ...
          get(boxes(k), 'YData'), ...
          colorsSpecies(end-k+1,:), ...
          'FaceAlpha', 0.5);
end
xlabel('Rodent species');
ylabel('Ectoparasite count per rodent');
title('Ectoparasite Load Across Rodent Species');
grid on;


%% 6. Comparing Ectoparasites Across Habitats

% Calculating descriptive statistics by habitat
habitatStats = groupsummary(parasiteModel, 'HBT_TYPE', ...
    {'mean', 'median', 'std', 'min', 'max'}, ...
    'Total_parasitical_load');

% Calculating zero counts and proportions
habitatZeros = groupsummary(parasiteModel, 'HBT_TYPE', ...
    {'sum', 'mean'}, 'ZeroParasites');

% Converting zero proportions to percentages
habitatZeros.mean_ZeroParasites = ...
    habitatZeros.mean_ZeroParasites * 100;

% Joining Tables
habitatSummary = join(habitatStats, ...
    habitatZeros(:, {'HBT_TYPE', 'sum_ZeroParasites', ...
    'mean_ZeroParasites'}), ...
    'Keys', 'HBT_TYPE');

% Displaying results
fprintf('\n');
disp("---------------------------------")
disp("Descriptive Statistics by Habitat")
disp("---------------------------------")
disp(habitatSummary);

% Creating a boxplot comparing parasite load across habitats
colors = [
    0.90 0.67 0.27;  % Agriculture
    0.41 0.66 0.37;  % Grazing
    0.20 0.48 0.31;  % Natural
    0.40 0.54 0.77   % Village
];

figure;
boxplot(parasiteLoad, habitats, ...
    'Symbol', 'o', ...
    'OutlierSize', 3);

% Find box objects
boxes = findobj(gca, 'Tag', 'Box');

% MATLAB returns box objects in reverse order
for k = 1:length(boxes)
    patch(get(boxes(k), 'XData'), ...
          get(boxes(k), 'YData'), ...
          colors(end-k+1,:), ...
          'FaceAlpha', 0.5);
end
xlabel('Habitat type');
ylabel('Ectoparasite count per rodent');
title('Ectoparasite Load Across Habitats');
grid on;


%% 7. Comparing Ectoparasites Across Sex

% Calculating descriptive statistics for each sex
sexStats = groupsummary(parasiteModel, 'SEX', ...
    {'mean', 'median', 'std', 'min', 'max'}, ...
    'Total_parasitical_load');

% Calculating zero counts and proportions
sexZeros = groupsummary(parasiteModel, 'SEX', ...
    {'sum', 'mean'}, 'ZeroParasites');

% Converting zero proportions to percentages
sexZeros.mean_ZeroParasites = ...
    sexZeros.mean_ZeroParasites * 100;

% Joining tables
sexSummary = join(sexStats, ...
    sexZeros(:, {'SEX', 'sum_ZeroParasites', ...
    'mean_ZeroParasites'}), ...
    'Keys', 'SEX');

% Displaying the results
fprintf('\n');
disp("------------------------------------")
disp("Descriptive Statistics by Rodent Sex")
disp("------------------------------------")
disp(sexSummary);

% Creating a boxplot comparing parasite load across rodent's sex
figure;

% Colours for rodents' sex
colorsSpecies = [
    0.20 0.50 0.85;  % Male - Blue
    0.70 0.40 0.80   % Female - Purple
];

boxplot(parasiteLoad, sex, ...
    'Symbol', 'o', ...
    'OutlierSize', 3);
% Find the box objects
boxes = findobj(gca, 'Tag', 'Box');

% Colour each box
for k = 1:length(boxes)
    patch(get(boxes(k), 'XData'), ...
          get(boxes(k), 'YData'), ...
          colorsSpecies(end-k+1,:), ...
          'FaceAlpha', 0.5);
end
xlabel('Rodent sex');
ylabel('Ectoparasite count per rodent');
title('Ectoparasite Load by Sex');
grid on;


%% 8. Comparing Ectoparasite load by breeding status

% Calculating descriptive statistics by breeding status
breedingStats = groupsummary(parasiteModel, 'BRS', ...
    {'mean', 'median', 'std', 'min', 'max'}, ...
    'Total_parasitical_load');

% Calculating zero counts and proportions
breedingZeros = groupsummary(parasiteModel, 'BRS', ...
    {'sum', 'mean'}, 'ZeroParasites');

% Converting proportions to percentages
breedingZeros.mean_ZeroParasites = ...
    breedingZeros.mean_ZeroParasites * 100;

% Joining tables
breedingSummary = join(breedingStats, ...
    breedingZeros(:, {'BRS', 'sum_ZeroParasites', ...
    'mean_ZeroParasites'}), ...
    'Keys', 'BRS');

% Displaying the results
fprintf('\n');
disp("-----------------------------------------")
disp("Descriptive Statistics by Breeding Status")
disp("-----------------------------------------")
disp(breedingSummary);

% Creating a boxplot comparing parasite load by breeding status
figure;

% Colours for breeding
colorsSpecies = [
    0.85 0.20 0.20;  % Breeding - Red
    0.30 0.65 0.35;  % Not Breeding - Green
];

boxplot(parasiteLoad, breeding, ...
    'Symbol', 'o', ...
    'OutlierSize', 3);
% Find the box objects
boxes = findobj(gca, 'Tag', 'Box');

% Colour each box
for k = 1:length(boxes)
    patch(get(boxes(k), 'XData'), ...
          get(boxes(k), 'YData'), ...
          colorsSpecies(end-k+1,:), ...
          'FaceAlpha', 0.5);
end
xlabel('Breeding status');
ylabel('Ectoparasite count per rodent');
title('Ectoparasite Load by Breeding Status');
grid on;


%% 9. Investigating Total body length 

% Calculating descriptive statistics for total length
meanTL = mean(totalLength);
medianTL = median(totalLength);
stdTL = std(totalLength);
minTL = min(totalLength);
maxTL = max(totalLength);

% Displaying the results
fprintf('\n');
disp("----------------------------")
disp("Total Body Length Statistics")
disp("----------------------------")
fprintf("Mean: %.2f\n", meanTL);
fprintf("Median: %.2f\n", medianTL);
fprintf("Standard deviation: %.2f\n", stdTL);
fprintf("Minimum: %.2f\n", minTL);
fprintf("Maximum: %.2f\n", maxTL);

% Creating a scatterplot
figure;
scatter(totalLength, parasiteLoad, 20, 'filled');
xlabel('Total body length (mm)');
ylabel('Ectoparasite count per rodent');
title('Relationship Between Body Length and Ectoparasite Load');
grid on;

% Calculating Spearman correlation between total length and parasite load
[spearmanTL, pValueTL] = corr(totalLength, parasiteLoad, ...
    'Type', 'Spearman', ...
    'Rows', 'complete');

% Displaying the results
fprintf('\n');
disp("-------------------------------------------------")
disp("Correlation Between Body Length and Parasite Load")
disp("-------------------------------------------------")
fprintf("Spearman correlation coefficient: %.3f\n", spearmanTL);
fprintf("P-value: %.4f\n", pValueTL);


%% 10. Investigating Species Distribution Across Habitats

% Counting the number of rodents of each species in each habitat
speciesHabitatTable = groupsummary(parasiteModel, ...
    {'SPECIES', 'HBT_TYPE'});

% Displaying the results
fprintf('\n');
disp("------------------------------------")
disp("Species Distribution Across Habitats")
disp("------------------------------------")
disp(speciesHabitatTable);


%% 11. Combining the Species and Habitat and analysing it

% Calculating parasite-load statistics for each combination
speciesHabitatStats = groupsummary(parasiteModel, ...
    {'SPECIES', 'HBT_TYPE'}, ...
    {'mean', 'median', 'std'}, ...
    'Total_parasitical_load');

% Displaying the results
fprintf('\n');
disp("------------------------------------")
disp("Parasite Load by Species and Habitat")
disp("------------------------------------")
disp(speciesHabitatStats);


%% 12. Exploring the Relationship Between Locality and Habitat

% Creating a table containing both variables
localityData = table(locality, habitats);

% Counting observations for each locality-habitat combination
localityHabitatCounts = groupsummary( ...
    localityData, {'locality', 'habitats'});

% Displaying the results
fprintf('\n');
disp("-----------------------------");
disp("Locality-Habitat Distribution");
disp("-----------------------------");
disp(localityHabitatCounts);


%% 13. Creating a Locality-Habitat Contingency Table

% Creating a contingency table
localityHabitatTable = ...
    groupcounts(table(locality, habitats), ...
    {'locality', 'habitats'});

% Displaying the results
fprintf('\n');
disp("----------------------------------");
disp("Locality-Habitat Contingency Table");
disp("----------------------------------");
disp(localityHabitatTable);


%% 14. Scatter Plot: Parasite Load vs Total Length by Species

% Identifying unique species
uniqueSpecies = categories(species);

% Creating a figure with four separate plots
figure;
tiledlayout(2, 2);

% Plotting each species separately
for i = 1:numel(uniqueSpecies)

    % Identifying observations belonging to this species
    currentSpecies = uniqueSpecies{i};

    % Finding observations belonging to this species
    speciesIndex = species == currentSpecies;

    % Selecting the next subplot
    nexttile;

    % Plotting total length against parasite load
    scatter( ...
        totalLength(speciesIndex), ...
        parasiteLoad(speciesIndex), ...
        25, 'filled', ...
        'MarkerFaceAlpha', 0.5);

    % Labelling the axes
    xlabel('Total Body Length (TL)');
    ylabel('Parasite Load');
    title(currentSpecies, 'Interpreter', 'none');
    ylim([0, max(parasiteLoad) * 1.05]);
    grid on;
end

% Adding an overall figure title
sgtitle('Relationship Between Body Length and Parasite Load by Species');


