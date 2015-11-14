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


    %randomly initialize C the covariance matrix to USU^T, but NEVER
    %actually store it! only store updates to U and S!

    if(size(X) ~= [77760, 149])           %obviously change for B dataset
       size(X)
       error('IPCA: bad input');
    end
    for i = 1:iters
        fprintf('----iteration %d\n', i);
        X(:,randperm(size(X,2)));         %good practice to shuffle:
        for t = 1:n; %CHANGE TO n LATER
            x = X(:, t);
            x_hat = U'*x;
            r = x - U*x_hat;
%             size(x_hat)
%             size(r)
%             size(S)
            
            
           %find eigendecomposition of the 2x2 matrix
           r_mag = norm(r);
           Q = [S + x_hat*x_hat', r_mag*x_hat; r_mag*x_hat', r_mag^2];
           [U_tilde, S_prime] = eig(Q);
           S = S_prime(1:(end-1), 1:(end-1));
           U = [U, r/r_mag]*U_tilde;
           if (size(U, 1) > k)
               %truncate
               U = U(:, 1:k);
           end
          
            
        end
        
    end

%C_hat +xx^T = [u r/r_norm]*[S +x_hat*x_hat^t, rnorm*xhat; rnorm*xhat,
%r_norm^2] * [U^t; r/r_nrom


end

