% PRINcipal COMPonent calculator
%   Calculates the principal components of a collection of points.
% Input:
%   X - D-by-N data matrix of N points in D dimensions.
% Output:
%   PCs - A matrix containing the principal components of the data.

function [PCs] = princomp(X)

    [height, width] = size(X);
    X_demean = X-repmat(mean(X,2), 1, width);
    
    [U,S,V] = svd(X_demean);
    % Since each column of X is a data point
    % The covaraince matrix is A*A'
    % and the columns of U is the eigen vectors for A*A'
    % Therefore we return U.
    PCs = U;
    % Note that the matlab pca function assume each row to be a data point
    
    
end