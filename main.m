addpath(genpath(pwd));
close all;

train_samples = 1000;
test_samples = 1000;
% Generate Data
train_data = rand(train_samples,2) .* 0.5;
test_data = (rand(test_samples,2) .* 0.5) + (rand(1) * 0.25);

train_label = ones(train_samples,1);
test_label = -ones(test_samples,1);


% Train SVM Model
cmd = '-s 2 -t 2';
m = svmtrain( train_label, sparse(train_data), cmd);

% Test SVML Model
[predictions, accuracy, prob_estimates] = ...
    svmpredict(test_label, sparse(test_data), m);
        
hold on
% Scatter Train and Test Data
scatter(train_data(:,1), train_data(:,2), 30, 'b'); 
scatter(test_data(:,1), test_data(:,2), 30, 'g'); 

% Scatter the Support Vectors
scatter(m.SVs(:,1), m.SVs(:,2), 20, 'r');