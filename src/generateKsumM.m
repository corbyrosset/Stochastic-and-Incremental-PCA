function [varargout]=generateKsumM(varargin)
% function [x]=generateKsumM(varargin)
% this function generates 3 random numbers for which the sum is equal to a
% certain value specified M
%
% INPUT ARGUMENTS:
% varargin{1}: n = integer number of rows          [Optional, default n = 10]
% varargin{2}: M = total sum of the random numbers [Optional, default M = 1.0]
%              M > 0, M < 0; can be either positive or negative
%              M can be also a COMPLEX number
%              M ~=0;
%
% varargin{3}: K = number of random numbers which  [Optional, default K = 3]
%
% OUTPUT ARGUMENTS:
% varargout{1}: x = matrix which stores all the random numbers 
%              { x is [n x K] | sum(x,2) = M }
%
%   Example:
%
%     x = generateKsumM;
%     plot(sum(x,2),'b.-');
% 
% 
% Developed by
% Mario Castro Gama
% MSc hydroinformatics UNESCO-IHE
% 2011.05.17
%
% Last Update
% 2014.02.06, changed to varargin and varargout.
%
  switch nargin
    case 0; % run example
      n = 10;
      total_sum = 1;
      K = 3;
      
    case 1; % just the length is provided and the sum is 1.0;
      if ~isempty(varargin{1});
        if ((mod(varargin{1},1)==0) && (varargin{1}>0)); 
          n = varargin{1};
        else
          error('  ErrorInput(1):  n = ~Integer or n<0'); return;
        end
      else
        error('  ErrorInput(1):  n = [], not provided'); return;
      end
      total_sum = 1;
      K = 3;
    
    case 2; % given also the value of the TOTAL_SUM of the random numbers
      if ~isempty(varargin{1}); % check n
        if ((mod(varargin{1},1)==0) && (varargin{1}>0)); 
          n = varargin{1};
        else
          error('  ErrorInput(1):  n = ~Integer or n<0'); return;
        end
      else
        error('  ErrorInput(1):  n = [], not provided'); return;
      end
      
      if ~isempty(varargin{2}); % check M
        if varargin{2} ~=0;
          total_sum = varargin{2};
        else
          error(' ErrorInput(2):  M = 0, not allowed'); return;
        end
      else
        total_sum = 1;
        disp('  Warning(2):  TOTAL_SUM of the random numbers not provided, set to 1.0');
      end
      K = 3;
    
    case 3; %give also the quantity K of random numbers which add up to TOTAL_SUM
      if ~isempty(varargin{1}); % check n
        if ((mod(varargin{1},1)==0) && (varargin{1}>0)); 
          n = varargin{1};
        else
          error('  ErrorInput(1):  n = ~Integer or n<=0'); return;
        end
      else
        error('  ErrorInput(1):  n = [], not provided'); return;
      end
      
      if ~isempty(varargin{2});
        if varargin{2} ~=0;
          total_sum = varargin{2};
        else
          error(' ErrorInput(2):  M = 0, not allowed'); return;
        end
      else
        total_sum = 1;
        disp('  Warning(2):  TOTAL_SUM = [] of random numbers not provided, set TOTAL_SUM = 1.0');
      end
      
      if ~isempty(varargin{3});
        if ((mod(varargin{3},1)==0) && (varargin{3}>1)); % lower limit is 2 random numbers
          K = varargin{3};
        else
          error('  ErrorInput(3):  K = ~Int or K < 2'); return;
        end
      else
        K = 3;
        disp('  Warning(3):  K = []; number of random generated not provided, set K = 3');
      end
    otherwise
      error(' ErrorInput(4):  Too many input arguments'); return;
  end
  
  % create the random numbers matrix x
  x = zeros(n,K);
  for ii=1:n;
    r1 = rand(1,K-1);
    r1 = sort(r1); % this is the key of the algorithm
    for jj =1:K;
      switch jj
        case 1; 
          x(ii,1) = total_sum * r1(1);
        case K;
          x(ii,K) = total_sum * (1 - r1(end));
        otherwise
          x(ii,jj) = total_sum * (r1(jj)-r1(jj-1));
      end
    end
  end
  varargout{1} = x;
end