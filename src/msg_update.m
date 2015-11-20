%% Performs the "positive" Frobenius-MD update. The iterate M is:
%     M = U * diag( S ) * U'
%
% k - the dimension of the subspace which we seek
% U, S - "nontrivial" eigenvectors and eigenvalues of the iterate
% eta - the step size
% x - the sample vector
% epsilon - threshold for rank1update and msgproject
%%

function [U,S]=msg_update(k,U,S,eta,x,epsilon)

d=size(U,1);
b=1/k;
value = 0;
[U,S]=rank1update(U,S,value,eta,x,epsilon);
[U,S]=msgproject(U,S,value,d,b,epsilon);
end

%% Performs the rank-1 update:
%     M + eta * sample * sample'
% where M is represented as a partial eigendecomposition:
%     M = U * diag( values ) * U' + ( I - U * U' ) * value
%
% U, values - "nontrivial" eigenvectors and eigenvalues of M
% value - the (repeated) eigenvalue in the subspace orthogonal to U
% eta - the scale of the rank-1 update
% sample - the vector of the rank-1 update
% epsilon - threshold for adding a new "nontrivial" dimension
%%

function [U,S]=rank1update(U,S,value,eta,x,epsilon )
[d, k] = size(U);
xhat=U'*x;
res=x-U*xhat;
resnorm=norm(res);
if ((resnorm < epsilon) || (k >= d))
    dS=diag(S)+eta*(xhat*xhat');
    dS=0.5*(dS+dS'); %**make sure matlab knows the matrix is real symmetric
    
else
    U=[U,res/resnorm ];
    dS=[diag(S)+eta*(xhat*xhat'), (eta*resnorm)*xhat;...
        (eta*resnorm)*xhat', value+(eta*resnorm*resnorm)];
    dS=0.5*(dS+dS'); %**make sure matlab knows the matrix is real symmetric
end
[Utilde,newS]=eig(dS);
U=U*Utilde;
S=diag(newS)';
U=real(U);
S=real(S);
end


%% Projects an iterate onto the constraints with respect to Frobenius norm.
%
% KK_max - a hard cap on the dimension of the iterate
% U, values - "nontrivial" eigenvectors and eigenvalues of the iterate
% value - the (repeated) eigenvalue in the subspace orthogonal to U
% KK_max - a hard cap on the dimension of the iterate
% dd - the dimension of the overall space
% bb - upper bound on maximum eigenvalue
% epsilon - threshold for projection_shift
%%

function [U,S] = msgproject(U,S,value,d,b,epsilon)

k=length(S);
S2=[ S, value ];
c=[ones(1,k),d-k];
shift=projshift(S2,c,b,epsilon);
S=S+shift;
S(S < 0 )=0;
S(S > b )=b;
indices=(S ~= value );
U=U(:,indices );
S=S(indices);

% renormalize the vectors, just to be sure
U=U*pinv(sqrtm(U'*U));
U=real(U);
end

%% Finds a shift SS such that:
%     shift = sum_i(c[i] * max(0,min(b,S[i]+shift)))
%
% S - vector of eigenvalues
% c - vector of eigenvalue multiplicities
% b - upper bound on maximum eigenvalue
% epsilon - threshold for error checking
%%

function shift=projshift(S,c,b,epsilon)

if(size(S,1) ~= 1)
    error('projshift: eigenvalues must be a row vector' );
end
if(size(c,1 ) ~= 1)
    error('projshift: multiplicities must be a row vector' );
end
if(length(S) ~= length(c))
    error('projshift: eigenvalues and multiplicities must be same length');
end

k=length(S);
d=sum(c);

if((b*d <= 1) || (b > 1))
    error('projshift: bound must be in the range (1/d,1]');
end

[S,idx]=sort(S);
c=c(idx);

sums=[0,cumsum(S.*c)];
counts=[0,cumsum(c)];

shifts=(1-(repmat(sums,k+1,1)-repmat(sums',1,k+1))-...
    b*(d-repmat(counts,k+1,1)))./(repmat(counts,k+1,1)-...
    repmat(counts',1,k+1));

error1=-(repmat([S,inf ],k+1,1)+shifts-b); error1(error1 < 0)=0;
error2=(repmat([-inf,S ],k+1,1)+shifts-b); error2(error2 < 0)=0;
error3=-(repmat([S,inf ]',1,k+1)+shifts);  error3(error3 < 0)=0;
error4=(repmat([-inf,S ]',1,k+1)+shifts);  error4(error4 < 0)=0;

condition=error1+error2+error3+error4;
shift=shifts(condition == (min(min(condition))));

if(length(shift) < 1)
    error('projshift: solution not found');
end
shift=shift(1);

result=S+shift;
result(result < 0)=0;
result(result > b)=b;
total=sum(result .* c);
if(abs(total-1) > epsilon)
    error('projshift: solution does not sum to 1 (%f)',total);
end
end