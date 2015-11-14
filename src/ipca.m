%% Incremental PCA

% update of the form C^(t) = P_{rank-k}(C^(t-1) + x*x^T)
% but stored in the form of an singular value decomposition of the 
% covariance (2nd moment) matrix as so:
%

function U = ipca(X)

    X = X';               % to make things work
    iters = 1;            % how many times to loop over entire training set
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
    k = n;                % MUST conform to api: return a dxn matrix
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace, d by k
    S = diag(randn(k,1)); % C^0 = USU^T random eigendecomposition, k by k

%randomly initialize USU^T
S = diag(randn(n,1)); % Random negative eigenvalues
V = orth(randn(n)); % Random unitary matrix
%randomly initialize C the covariance matrix

%C_hat +xx^T = [u r/r_norm]*[S +x_hat*x_hat^t, rnorm*xhat; rnorm*xhat,
%r_norm^2] * [U^t; r/r_nrom


end

