addpath(genpath(pwd));
close all;

vector_length = 2;
train_samples = 1000;
test_samples = 1000;
% Generate Data
train_data = rand(train_samples,vector_length) .* 0.5;
test_data = (rand(test_samples,vector_length) .* 0.5) + (rand(1) * 0.25);

train_label = ones(train_samples,1);
test_label = -ones(test_samples,1);



% Train SVM Model
cmd = '-s 2 -t 2';
m = svmtrain( train_label, sparse(train_data), cmd);

% Test SVML Model
[predictions, accuracy, prob_estimates] = ...
    svmpredict(test_label, sparse(test_data), m);


%[pc,score,latent,tsquare] = princomp(train_data);
%disp(cumsum(latent)./sum(latent));

hold on
if vector_length >= 3
% Scatter Train and Test Data
scatter3(train_data(:,1), train_data(:,2),train_data(:,3), 'b'); 
scatter3(test_data(:,1), test_data(:,2),test_data(:,3),  'g'); 

% Scatter the Support Vectors
scatter3(m.SVs(:,1), m.SVs(:,2),m.SVs(:,3),  'r');
elseif vector_length == 2
scatter(train_data(:,1), train_data(:,2), 'b'); 
scatter(test_data(:,1), test_data(:,2),  'g'); 

% Scatter the Support Vectors
scatter(m.SVs(:,1), m.SVs(:,2),  'r');   
end