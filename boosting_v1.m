%% first sucessful try
% no location data
temp_data = reshape(data2, 145, 145, 200);
conv_data = zeros(145, 145, 200);
a = zeros(145, 145);
a(1:2,1:2) = 1/4;
A = fft2(a);
for i=1:200
    conv_data(:,:,i) = ifft2(fft2(temp_data(:,:,i)).*A);
end

% One 
conv_data=reshape(conv_data, 145*145, 200);



K = 17;
results = zeros(size(conv_data, 1), 1);
black_vec = ones(200, 1)*-1000;
black_list = false(size(conv_data, 1), 1);
% The first iterations
% C is D by K
[C, labels] = km(conv_data', K, 1000, initMeans(:, 1:200)');
variance = zeros(17, 1);
for j=1:K
%    count = sum(labels==j);
%     temp = conv_data(labels==j, :) - repmat(C(:,j)',count, 1);
%     var(j) = norm(temp)/count;
    variance(j) = norm(var(conv_data(labels==j,:), 0, 1));
end



[min_var,min_idx] = min(variance);
disp(sprintf('minimum variance %f, group %i', min_var, min_idx));
results(labels==min_idx) = 1; % First finalized cluster, so WOLG, we name it 1 
%number of elements in the Minimum-variance-Cluster(MC)
mc_count = sum(labels==min_idx);
disp('here')
conv_data(labels==min_idx,:)=repmat(black_vec', mc_count,1);
disp('here2')
black_list(labels==min_idx)=true;
%%
% The next K-1 iterations
for i=1:2
    % Find the smallest variance
    initMeans = zeros(K-i+1, 200);
    initMeans(1,:) = black_vec; % This so that we know which black cluster is
    % TODO: initialize the other initial means
    
    initMeans_idx = randsample(find(black_list==0), K-i);
    initMeans(2:end,:) = conv_data(initMeans_idx, :);
    
    [C, labels] = km(conv_data', K-i+1, 1000, initMeans');
    variance = zeros(K-i+1,1);
    for j=1:(K-i+1)
%         count = sum(labels==j); % Number of elements in cluster j
%         temp = conv_data(labels==j, :) - repmat(C(:,j)',1,count);
%         variance(j) = norm(temp)/count;
        variance(j) = norm(var(conv_data(labels==j,:), 0, 1));
    end
    
    [min_var,min_idx] = min(variance(2:end)); % Excluding the first black cluster;
    % Now min_idx is the cluster that is not black, that has the smallest
    % var
    disp(sprintf('minimum variance %f, group %i', min_var, min_idx));
    results(labels==min_idx) = i+1; % A different group
    %number of elements in the Minimum-variance-Cluster(MC)
    
    mc_count = sum(labels==min_idx);
    conv_data(labels==min_idx, :)=repmat(black_vec', mc_count,1);
    
    
end

