%% 1. Importing prof Matthee's raw data into MATLAB

dataFile = "Biomathematics_synthetic.xlsx";
parasiteData = readtable(dataFile);

% Checking out how the table was imported in
head(parasiteData)


%% 2. Creating the variables I want to work with

variables = {'HBT_TYPE' 
    'LOCALITY'          % Added the variable last minute to check if locality had any effects
    'SPECIES' 
    'SEX' 
    'BRS' 
    'TL' 
    'Total_parasitical_load'};
disp(variables) % Checking it worked

% Extracting the needed columns
parasiteModel = parasiteData(:, variables);
head(parasiteModel) % Checking it works


%% 3. Defining the types of variables 

% Categorical variables
parasiteModel.HBT_TYPE = categorical(parasiteModel.HBT_TYPE);
parasiteModel.SPECIES = categorical(parasiteModel.SPECIES);
parasiteModel.SEX = categorical(parasiteModel.SEX);
parasiteModel.BRS = categorical(parasiteModel.BRS);
parasiteModel.LOCALITY = categorical(parasiteModel.LOCALITY); % Added last minute

% Making sure there are only two categories for sex and not 4
parasiteModel.SEX = categorical( ...
    upper(string(parasiteModel.SEX)));

 % Checking the categorical data worked
 disp(categories(parasiteModel.HBT_TYPE));
 disp(categories(parasiteModel.SPECIES));
 disp(categories(parasiteModel.SEX));
 disp(categories(parasiteModel.BRS));
 disp(categories(parasiteModel.LOCALITY));

% Numerical variables 
% Checking variable types
class(parasiteModel.TL) % Cell
class(parasiteModel.Total_parasitical_load) % Double

% Convert TL only if stored as text
if ~isnumeric(parasiteModel.TL)
    parasiteModel.TL = str2double(parasiteModel.TL);
end
class(parasiteModel.TL) % Double


%% 4. Data Validation 
%{ 
Checking if any of the data is missing a value so I could fix it in the
excel spreadsheet or so i could query it. I also check the numerical data
here
%}

missingValues = sum(ismissing(parasiteModel));
% Creating a summery table 
missingValueTable = table( ...
    parasiteModel.Properties.VariableNames', ...
    missingValues', ...
    'VariableNames', {'Variable', 'MissingValues'});

% Display results
disp("Missing values per variable:")
disp(missingValueTable)

% Checking for invalid total lengths
invalidTL = sum( ...
    ~isfinite(parasiteModel.TL) | parasiteModel.TL <= 0);

% Check for invalid parasite counts
invalidParasiteLoad = sum( ...
    ~isfinite(parasiteModel.Total_parasitical_load) | ...
    parasiteModel.Total_parasitical_load < 0 | ...
    parasiteModel.Total_parasitical_load ~= ...
    floor(parasiteModel.Total_parasitical_load));

% Display results
fprintf("Invalid total lengths: %d\n", invalidTL);
fprintf("Invalid parasite counts: %d\n", invalidParasiteLoad);

% Saving the checked and prepared dataset
save("parasiteModel.mat", "parasiteModel");


