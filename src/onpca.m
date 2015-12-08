function U = onpca(X, k)
    X = X';               % to make things work
    iters = 5;            % how many times to loop over entire training set
    t = 0;                % iterate
    n = size(X, 2);       % number of examples
%     k = 200;              % MUST conform to api: return a dxn matrix
    d = size(X, 1);       % dimensionality
    U = orth(rand(d, k)); % randomly init learned subspace
    eta = 1;              % learning rate, set to what??
    
    M = eye - U*U';
    
    if(size(X, 1) ~= 32256)          %obviously change for B dataset
       size(X)
       error('Online PCA: bad input');
    end
    h = waitbar(0,'Waiting for Online PCA...');

    for i = 1:iters
        fprintf('----iteration %d\n', i);
        X(:,randperm(size(X,2)));         %good practice to shuffle:
        for t = 1:n;
            x = X(:, t);
            
            temp1 = logm(M) - eta*x*x';
            temp2 = expm(temp1);
            M = temp2;
            waitbar((n*(i-1) + t)/(iters*n),h)   
        end 
    end 
    close(h);
    

end