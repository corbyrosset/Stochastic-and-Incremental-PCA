function U = msg(X)

    X = X';               % to make things work
    iters = 3;            % how many times to loop over entire training set
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
    k = n;                % MUST conform to api: return a dxn matrix
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace, d by k
    S = diag(randn(k,1)); % C^0 = USU^T random eigendecomposition, k by k
    
end

