function U = svdpca(X, k)
    X = X';               % to make things work
    n = size(X, 1);       % number of examples
    d = size(X, 2);       % dimensionality
    %size(X)
    %avg = (mean(X, 1))';
    %X = X - (repmat(avg, 1, n))';
    [U, S, V] = svd(X);
    %size(U)
    [S, indices] = sort(diag(S), 'descend');
    U = U(:, indices);
    
    
end