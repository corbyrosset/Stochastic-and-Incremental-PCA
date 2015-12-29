%% Incremental PCA

% update of the form C^(t) = P_{rank-k}(C^(t-1) + x*x^T)
% but stored in the form of an singular value decomposition of the 
% covariance (2nd moment) matrix as so:
%

function U = ipca(X, k)
    X = X';               % to make things work
    iters = 1;            % how many times to loop over entire training set
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace, d by k
    S = diag(randn(k,1)); % C^0 = USU^T random eigendecomposition, k by k

    if(size(X, 1) ~= 32256)          
       size(X)
       error('IPCA: bad input');
    end
    h = waitbar(0,'Initializing waitbar...');
    for i = 1:iters
        fprintf('----iteration %d\n', i);
        X(:,randperm(size(X,2)));         %good practice to shuffle:
        for t = 1:n; 
           x = X(:, t);
           x_hat = U'*x;
           r = x - U*x_hat;
           r_mag = norm(r);
           Q = [S + x_hat*x_hat', r_mag*x_hat; r_mag*x_hat', r_mag^2];
           if (r_mag == 0) %strage edge case, ignore
              display('r_mag is zero, all black column?'); 
              continue;
           end
           
           %sanity checking
           if (sum(sum(isnan(Q))))
               error('Q has NaNs in it');
           end
           if (sum(sum(isinf(Q))))
               error('Q has Infs in it');
           end
           
           [U_tilde, S_prime] = eig(Q); 
           S = S_prime;
           U = [U, r/r_mag]*U_tilde;
           
           %sort by decending eigenvalues
           [S_blah, ind] = sort(diag(S), 'descend');
           S = diag(S_blah);
           U = U(:, ind);
           if (size(U, 1) > k)
               %truncate
               U = U(:, 1:k);
               S = S(1:k, 1:k);
           end
           waitbar((n*(i-1) + t)/(iters*n),h) 
        end   
    end
    maxevalue = max(diag(S)); 
    domain = [1:k];
    scatter(domain, diag(S)); hold on;
    title('eigenvalues of ipca'); %plot normalized eigenvalues
    hold off;
    close(h);
end

