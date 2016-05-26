%% K-Means
%  Separate data points into K clusters with no other information.
% Inputs:
%  A - D-by-N matrix of N points in D dimensions.
%  K - Integer number of clusters to detect. 
%  maxiter - maximum number of iterations (pick maybe 100, or 1000)
% Outputs:
%  C - D-by-K matrix with the learned cluster centroids.
%  labels - Length N vector with integer (1, 2, ..., K) class assignments.

function [C, labels] = km_noinit(A, K, maxiter)

labels = randi([1,K], size(A, 2), 1);
old_labels = labels;
C = zeros(size(A,1), K);
dist = zeros(1,K);
for i = 1:maxiter
    for j=1:K
        C(:,j) = mean(A(:,labels==j), 2);    
    end
    for l = 1:size(A, 2)
        for k = 1:K
            dist(k) = norm(A(:,l)-C(:,k));
        end
        [temp, labels(l)] = min(dist);
        
    end
    
    if sum(labels ~= old_labels) == 0
        disp('holy moly');
        break
    end
    
    old_labels = labels;
    fprintf('iter %i',i);
end


end
