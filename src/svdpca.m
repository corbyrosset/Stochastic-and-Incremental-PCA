function [U, S] = svdpca(X, k)
    X = X';               % to make things work
    
    [U, S, ~] = svd(X, 'econ');
    %remember, X*X^t = U*S^2*U^T for X = U*S*V^T
    
    [S, indices] = sort(diag(S), 'descend');
    U = U(:, indices);
    U = U(:, 1:k);
    S = S(1:k, 1);
end