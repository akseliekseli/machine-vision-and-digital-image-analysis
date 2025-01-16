%% MVIDA Exercise 1
% Akseli Suutari

%% Task 1
clc, clearvars, close all

fixed = imread('ex21a.tif'); fixed = fixed(:,:,1);
moving = imread('ex21b.tif'); moving = moving(:,:,1);

%cpselect(moving, fixed)
%save("ctrl_points.mat", 'fixedPoints', "fixed", "movingPoints", "moving")

%% 
clc, clearvars, close all
load("ctrl_points.mat")

T = fitgeotform2d(movingPoints,fixedPoints, "lwm", 6);

transformed = imwarp(moving, T);

imshowpair(fixed, transformed)


%% Task 2 Bayesian Classifier
clc, clearvars, close all

load 'Data for Exercise 1.mat'

features = data(:, 1:2);
labels = data(:, 3);
posterior = [];
% Create the posterior distributions
for label=0:1
    class = features(labels == label, :);
    % Compute how big portion of the samples are which class
    prior = size(class, 1) / size(data, 1);

    % Fitting a Gaussian distribution for the classes to use in the
    % classification
    params = struct('mu', mean(class), 'sigma', cov(class));

    % Create a grid for decision boundary visualization
    x1_range = linspace(min(features(:, 1)), max(features(:, 1)), 100);
    x2_range = linspace(min(features(:, 2)), max(features(:, 2)), 100);
    [x1_grid, x2_grid] = meshgrid(x1_range, x2_range);
    grid_points = [x1_grid(:), x2_grid(:)];

    % Compute the posterior probabilities for each grid point
    posterior(:,end+1) = mvnpdf(grid_points, params.mu, params.sigma) * prior;
end
total_posterior = posterior(:,1) + posterior(:,2);

posterior(:,1) = posterior(:,1) ./ total_posterior;
posterior(:,2) = posterior(:,2) ./ total_posterior;

% Classify each grid point
decision_boundary = reshape(posterior(:,2) > posterior(:,1), size(x1_grid));

figure;
hold on;
scatter(features(labels==0, 1), features(labels==0, 2), 'ko',  'DisplayName', 'Salmon');
scatter(features(labels==1, 1), features(labels==1, 2), 'ro', 'DisplayName', 'Sea Bass');
contour(x1_grid, x2_grid, decision_boundary, [0.5 0.5], 'k', 'DisplayName', 'Decision Boundary');
legend;
title('Bayesian Classifier for Fish Classification');
xlabel('Lightness');
ylabel('Width');
grid on;
hold off;


%% Task 2 KNN
clc, clearvars, close all
load 'Data for Exercise 1.mat'

error_best = inf;
k_best = 1;
for k=1:20
    classifications = knn(data(:,3)', data(:,1:2)', data(:,1:2)', 6)
    error = sum(classifications~=data(:,3)')/length(data(:,3)')
    if error < error_best
        error_best = error;
        k_best = k;
    end
end
k_best
error_best


function C = knn(trainclass, traindata, data, k)
    % Data normalization and making the data so that columns are different 
    % features and rows are samples. 
    train_x = normalize(traindata');
    train_y = trainclass';
    test_x = normalize(data');

    % Calculate Euclidean distances between (test) data and training points
    euc_d = pdist2(train_x, test_x);
    
    % Loop the test samples:
    % 1. Sort the column of eucledian distances
    % 2. get the corresponding training classees
    % 3. take the k-nearest classes
    % 4. Assign the mode of the k nearest samples to be the class of the 
    %   test sample.
    C = [];
    for ii=1:length(test_x)
        euc_col = euc_d(:, ii);

        [sorted_d, idx] = sort(euc_col);

        sorted_y = train_y(idx);
        
        k_nearest_y = sorted_y(1:k);
        C(ii) = mode(k_nearest_y); 
    end
end

%% Task 2 Comparisation

% Choosing between the Bayesian classifier and KNN classifier comes down to 
% the characteristics of the dataset. With more separated classes the KNN works
% well since it is an simple model to implement. However with overlapping classes
% KNN loses accuracy.
% Bayesian classifier is a robust method for overlapping classes. However it
% might suffer with some distributions of the data. 

%% Task 2 MLP

% For this task the MPL should have two input nodes and two output nodes and some
% (fairly few) hidden nodes. Each datapoint will be fed to the MLP, possibly in
% batches, and the weights and biases of the nodes are tuned using backpropagation
% and gradient descent based on which of the output nodes has higher value i.e.
% which is the predicted class.

%% Task 2 SVM

% In this task a liner SVM could be used, since the data can be separated fairly
% well with a line.
% To do this, the data is transformed into a higher dimensional space using a kernel
% to get a better separation of the data. After that a linear hyperplane is created so
% that is maximizes the margin (distance between the closest datapoint, support vector, and
% the hyperplane).
% 
% The problem is an optimization problem.


%% Task 3 K-Means
clc, clearvars, close all

load iris_dataset.mat

% Make the irisTargets to be only one vector with classes 1, 2, 3
irisTargets(1,irisTargets(2,:)==1) = 2;
irisTargets(1,irisTargets(3,:)==1) = 3;
irisTargets = irisTargets(1,:);

ks = [2, 3, 4, 5]

for k=ks
    figure
    [b, Theta] = kmeans(irisInputs, k, 100)
    
    gplotmatrix(irisInputs', [], b)
    title(sprintf('k=%d', k))
    drawnow

end

% With k=3 it looks the best

function [b, Theta] = kmeans(X, k, n_iters)  
    % function [b, Theta] = kmeans(X, k)  
    
    M = size(X,1);
    N = size(X,2);
    b = zeros(1, N);

    % Initialize θ_j (j = 1, . . . , k) randomly.
    rng('default')
    temp = randperm(N);
    Theta = X(:, temp(1:k));
    iters=0;
    while iters <=n_iters
        iters = iters +1;
        Theta_old = Theta; 
        
        % Find nearest cluster j for sample i
        for ii=1:length(X)
            [~, b(ii)] = min(sum(sqrt((X(:,ii) - Theta_old).^2)));
        end
        % Update θ_j as the mean of samples where b(i) = j.
        for ii=1:k
            mean(X(:, b==ii),2)
            Theta(:,ii) = mean(X(:, b==ii), 2)
        end
        % Repeat until no change in any of θ_j between two successive iterations.
        if mean((Theta_old(:) - Theta(:)).^2) < 1e-6
            break
        end

    end
end