%% Matrix Stochastic Gradient Descent for PCA
% from "Stochastic Optimization of PCA with Capped MSG", Arora et al

% very similar to incremental pca, but with better theoretical bounds
% for convergence, and it is guaranteed not to get "stuck". 

% update of the form P^(t) = P_{trace(P) = k, P<= I}(P^(t-1) + \eta_t*x*x^T)
% but stored in the form of an singular value decomposition of the 
% covariance (2nd moment) matrix as so:

function U = msg(X, k)

    X = X';               % to make things work. X is matrix of [0, 255]
%     X = normc(X); %OH GOD
    iters = 7;              % how many times to loop over entire training set
    t = 0;                  % iterate
    n = size(X, 2);         % number of examples
    d = size(X, 1);         % dimensionality
    U = zeros(d, 1);        % provably better accuracy with all zeros initial
    S = zeros(1, 1);        % both U and S will grow larger...
    eta = sqrt(k/(n*iters));% seems to be good enough eta
    epsilon = 0.000001;         % don't know what the significance of this is...
    warning('off','all');   %because it clutters screen as matrix initializes
   
    if(size(X, 1) ~= 32256)           %obviously change
       size(X)
       error('IPCA: bad input');
    end
    h = waitbar(0,'Initializing waitbar...');
    for i = 1:iters
        fprintf('----iteration %d\n', i);
        X(:,randperm(size(X,2)));         %good practice to shuffle:
        for t = 1:n; 
           x = X(:, t);
           
           [U,S] = msg_update(k,U,S,eta,x,epsilon);
           if (sum(S) > 1)
               [U, S] = msgsample(k, U, S);   
           end
           if (rank(U) > k + 20) %why +20?, bc this happens quite often
               [U, S] = msgsample(k, U, S);
           end
           waitbar((n*(i-1) + t)/(iters*n),h);
        end
        
    end
    [U, S] = msgsample(k, U, S);
    domain = [1:k];
    scatter(domain, S); hold on;
    title('eigenvalues of msg');
    hold off;
    close(h);
    
end

function X = candN(X)
    mean = sum(X, 2)/size(X, 2);
    stdtrain = std(X');
    Xcenter = bsxfun(@minus, X, mean);
    X = bsxfun(@rdivide, Xcenter, stdtrain');
end

