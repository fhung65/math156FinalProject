%% THIS FILE WAS USED TO GENERATE THE FIRST SET OF BOOSTED results
% ONLY 12 clusters in the end were generated
% Others converged??????



load('Indian_pines_gt.mat')
load('Indian_pines_corrected.mat')
data = reshape(indian_pines_corrected, 145*145, 200);

%% Adding location feature
lambda = 60;
b = [1:145]';
temp = repmat(b, 1, 145);
a = reshape(temp', 145*145, 1);%x coord
b = repmat(b, 145, 1);% y coord
c = [a,b];
%%

data=  [data, c*lambda];
data2 = data(:,1:200);

%% 
ipc = indian_pines_corrected;
% sampleRange = 13*[1:10]; % 145/5 = 29
sampleRange = 29*[1:4]; % 145/5 = 29
% sampleY = repmat(sampleRange', 10, 1);
sampleY = repmat(sampleRange', 4, 1);


% sampleX = reshape(repmat(sampleRange', 1, 10)', 100, 1);
sampleX = reshape(repmat(sampleRange', 1, 4)', 16, 1);
samplePoints = [sampleY, sampleX];


initMeans = zeros(16,202);

%17th point
pt = [randi(145), randi(145)];
rngY = [pt(1)-2:pt(1)+2];
rngX = [pt(2)-2:pt(2)+2];
samples = reshape(ipc(rngY,rngX,:), 25,200);
avg = mean(samples,1);
initMeans(17,:) = [avg, lambda*pt(2), lambda*pt(1)];


% for i = 1:100
for i=1:16
    pt = samplePoints(i,:);
    rngY = [pt(1)-2:pt(1)+2];
    rngX = [pt(2)-2:pt(2)+2];
    samples = reshape(ipc(rngY,rngX,:), 25,200);
    avg = mean(samples,1);
    initMeans(i,:) = [avg, lambda*pt(2), lambda*pt(1)];
end



[C, labels] = km(data', 16, 1000, initMeans');

%% Using a lot of cluster centers first, then cluster the cluster centers
[C, labels] = km(data', 100, 1000, initMeans');
idx = 1:100;
sidx = randsample(idx, 21);
[C2, labels2] = km(C, 21, 100, C(:,sidx));


labels3 = labels2(labels);


%% SVD


[height, width] = size(data2');
data2_demean = data2'-repmat(mean(data2',2), 1, width);
[U,S,V] = svd(data2_demean);

eig_val = diag(S);


princ_data = data2*U(:,1:115);
princ_initMeans = initMeans(:,1:200)*U(:,1:115);

[C, labels] = km(princ_data', 100, 1000, princ_initMeans');

idx = 1:100;
sidx = randsample(idx, 21);
[C2, labels2] = km(C, 21, 100, C(:,sidx));
labels3 = labels2(labels);



%%
temp_data = reshape(data2, 145, 145, 200);
conv_data = zeros(145, 145, 200);
a = zeros(145, 145);
a(1:2,1:2) = 1/4;
A = fft2(a);
for i=1:200
    conv_data(:,:,i) = ifft2(fft2(temp_data(:,:,i)).*A);
end

% One 
% conv_data=reshape(conv_data, 145*145, 200);
% or the other, with position data
conv_data = [reshape(conv_data, 145*145, 200), c*lambda];


[C, labels] = km(conv_data', 17, 1000, initMeans');
% 
% idx = 1:100;
% sidx = randsample(idx, 21);
% [C2, labels2] = km(C, 21, 100, C(:,sidx));
% labels3 = labels2(labels);



%%
% First principal component
imagesc(reshape(princ_data(:,:,1), 145, 145))
imagesc(reshape(princ_data(:,:,2), 145, 145))




%% Boosting clustering

temp_data = reshape(data2, 145, 145, 200);
conv_data = zeros(145, 145, 200);
a = zeros(145, 145);
a(1:2,1:2) = 1/4;
A = fft2(a);
for i=1:200
    conv_data(:,:,i) = ifft2(fft2(temp_data(:,:,i)).*A);
end

% One 
% conv_data=reshape(conv_data, 145*145, 200);
conv_data_pos = [reshape(conv_data, 145*145, 200), data(:,end-1:end)];


K = 17;
results = zeros(size(conv_data_pos, 1), 1);
black_vec = ones(202, 1)*-1000; % size should be 202x1 if position data is used
black_list = false(size(conv_data_pos, 1), 1);
% The first iterations
% C is D by K
[C, labels] = km(conv_data_pos', K, 1000, initMeans');
variance = zeros(17, 1);
for j=1:K
%    count = sum(labels==j);
%     temp = conv_data(labels==j, :) - repmat(C(:,j)',count, 1);
%     var(j) = norm(temp)/count;
    variance(j) = norm(var(conv_data_pos(labels==j,:), 0, 1));
end



[min_var,min_idx] = min(variance);
disp(sprintf('minimum variance %f, group %i', min_var, min_idx));
results(labels==min_idx) = 1; % First finalized cluster, so WOLG, we name it 1 
%number of elements in the Minimum-variance-Cluster(MC)
mc_count = sum(labels==min_idx);
disp('here')
% Following line should use conv_data_pos if using pos data
conv_data_pos(labels==min_idx,:)=repmat(black_vec', mc_count,1);
disp('here2')
black_list(labels==min_idx)=true;
%%
% The next K-1 iterations
for i=1:16
    % Find the smallest variance
    initMeans = zeros(K-i+1, 202); % USE 202 if using pos data!
    initMeans(1,:) = black_vec; % This so that we know which black cluster is
    % TODO: initialize the other initial means
    
    initMeans_idx = randsample(find(black_list==0), K-i);
    initMeans(2:end,:) = conv_data_pos(initMeans_idx, :);
    
    [C, labels] = km(conv_data_pos', K-i+1, 1000, initMeans');
    variance = zeros(K-i+1,1);
    for j=1:(K-i+1)
%         count = sum(labels==j); % Number of elements in cluster j
%         temp = conv_data(labels==j, :) - repmat(C(:,j)',1,count);
%         variance(j) = norm(temp)/count;
        variance(j) = norm(var(conv_data_pos(labels==j,:), 0, 1));
    end
    [min_var,min_idx] = min(variance(2:end)); % Excluding the first black cluster;
    % Now min_idx is the cluster that is not black, that has the smallest
    % var
    disp(sprintf('minimum variance %f, group %i', min_var, min_idx));
    results(labels==min_idx) = i+1; % A different group
    %number of elements in the Minimum-variance-Cluster(MC)
    mc_count = sum(labels==min_idx);
    conv_data_pos(labels==min_idx, :)=repmat(black_vec', mc_count,1);
    figure
    imagesc(reshape(results, 145, 145));
    title(sprintf('iteration %i', i));
end
















