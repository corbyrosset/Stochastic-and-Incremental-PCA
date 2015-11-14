%% stochastic power iteration method from Arora, Cotter, Livescu, and Srebro
% input: X \in R^{d \times n} of n samples iid from an unknown data source
%   distribution. Each sample is of dimension d.
% output: U the decorrelated, orthogonal principle components describing 
%   the data. 
%
%updates: U^{t+1} = P_{orth}(U^t + \eta_t(x_t * x_t^T * U^{t -1}
%
%IMPLEMENTATION NOTES
% 1) by associativity, compute (x_t^T * U^{t -1}) first! O(dk) space
% 2) Projection onto the space of orthogonal matrices, while not necessary,
%    may be achieved by setting the update to be Q in the QR-factorization
%    of the the RHS using the built-in function: [Q,R]=qr(A)
% 3) because U and Q span the same subspace, perhaps wait until the end to
%    compute QR factorization...
% 4) >3 iterations through all examples are required. Why?
% 5) what to set eta to? Arora sets it to 1/sqrt(iteration#). Why?



  
function U = spm(X)
    X = X';               % to make things work
    iters = 3;            % how many times to loop over entire training set
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
    k = n;                % MUST conform to api: return a dxn matrix
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace
    eta = 1;              % learning rate, set to what??
    
    
    if(size(X, 1) ~= 32256)          %obviously change for B dataset
       size(X)
       error('SPA: bad input');
    end
    h = waitbar(0,'Initializing waitbar...');
    for i = 1:iters
        fprintf('----iteration %d\n', i);
        X(:,randperm(size(X,2)));         %good practice to shuffle:
        for t = 1:n;
            eta = 1/nthroot(i*n + t, 3);  %perhaps?
            x = X(:, t);
            U = U + eta*x*(x'*U);
            [U,~] = qr(U, 0);             %do more sparingly...
            waitbar((n*(i-1) + t)/(iters*n),h)
        end
        
    end
    

end

%% Manual QR just in case built-in tried to take matrix products...
function [Q,R] = QR(A)
    [m,n] = size(A);
    % compute QR using Gram-Schmidt
    for j = 1:n
       v = A(:,j);
       for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
       end
       R(j,j) = norm(v);
       Q(:,j) = v/R(j,j);
    end
end