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



  
function U = spm(X, Xdev, k)
    X = X';               % to make things work
    iters = 150;            % how many times to loop over entire training set
    i = 1;
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace
    etas = [10];           %can try others...            
    bestEta = 0;
    error = 10;
    tempError = 0;
    trainError = 10;
    
    if(size(X, 1) ~= 32256)         
       size(X)
       error('SPA: bad input');
    end
    for j = 1:length(etas)
        eta = etas(j);
        h = waitbar(0,'Waiting for Stochasic Power Method...');
        tempError = calcError(U, Xdev)
        while(i < iters && abs(error - tempError) > 0.5) %converged?
            fprintf('----iteration %d\n', i);    
            X(:,randperm(size(X,2)));                    %shuffle:
            for t = 1:n;
                eta = etas(j)/nthroot((i-1)*n + t, 2);   %good heuristic
                x = X(:, t);
                U = U + eta*x*(x'*U);
                if (mod(t, 5) == 0)
                    [U,~] = qr(U, 0);                    %do QR sparingly
                end
                if (mod(t, 300) == 0)
                   error = tempError;
                   tempError = calcError(U, Xdev);
                   trainError = calcError(U, X);
                   fprintf('train error: %d, dev error: %d, diff: %d, eta: %d\n', ...
                       trainError, tempError, abs(error - tempError), eta); 
                end
            end
            [U,~] = qr(U, 0);
            i = i + 1;
        end
        [U,~] = qr(U, 0);
    end
    close(h);
    

end


%% empirical error is defined to be reconstruction error
function obj = calcError(U, X) 
    obj = norm(X - U*(U'*X)); 
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