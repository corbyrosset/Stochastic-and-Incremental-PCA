%For now, we will use the small yale face dataset of 164 images total
%Later, we will use the Exteded Yale Face Database B:
%<http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html> which might
%actually be too big to fit in memory...

%Note it would be disastrous if at any point we were to store the
%covariance matrix. Instead, we will use computational tricks of outer
%products and the like to avoid it. 

%WARNING: the built-in pca() already centers and normalizes the data, doing
%so before calling pca() would result in an empty matrix!
%


function [] = RUNME()
%% load in raw data into 77760x164 matrix for Yale face data, 164 images
clear all;
close all;

bestKPCA = 0;
bestNPCA = 0;
testAccuracySubspacePCA = 0;
bestKSPM = 0;
bestNSPM = 0;
testAccuracySubspaceSPM = 0;
bestKIPCA = 0;
bestNIPCA = 0;
testAccuracySubspaceIPCA = 0;
bestKMSG = 0;
bestNMSG = 0;
testAccuracySubspaceMSG = 0;
bestKSPCA = 0;
bestNSPCA = 0;
testAccuracySubspaceSPCA = 0;
bestKSVDPCA = 0;
bestNSVDPCA = 0;
testAccuracySubspaceSVDPCA = 0;
global varspm;
varspm = 0;
global varipca;
varipca = 0;
global varmsg;
varmsg = 0;
global varsvdpca;
varsvdpca = 0;
global varspca;
varspca = 0;
global maxDimension; %VERY IMPORTANT, but only for sparse PCA...
maxDimension = 1200; %why this can't be on the above line is beyond me...
maxK      = 80; %VERY IMPORTANT must be significantly less than dimensionality
%because some algorithms run in O(d*k^2) PER EXAMPLE
global meantrain;
global meandev;
global meantest;
global dataVariance;


meantrain = [];
meandev = [];
meantest = [];



%%make sure to change to src directory!
% [train, trainlabels, test, testlabels] = readData('../data/yalefaces/yalefaces/');
[train, trainlabels, test, testlabels, dev, devlabels] = ...
        readYaleBPlusData('../data/CroppedYale/', 20, 20, 60);


%% plot training data onto top two principle components to "see" how good
%covariance matrix is 77760x77760 (45.1GB)

% U = pca(train');    %I think this uses stochastic pca anyway...
% top_2 = U(:, 1:2);  %first two columns are top two principle components
% C = top_2'*train;
% C = candN(C);       %we should really do this earlier
% C = C';
% scatter(C(:, 1), C(:, 2), 9, trainlabels);


%% train KNN model on data projected onto learned subspace
% train = (1/255)*train;
% dev = (1/255)*dev;
% test = (1/255)*test;

[train, meantrain] = candN(train);
[dev, meandev] = candN(dev);
[test, meantest] = candN(test);

dataVariance = calcVariance([], train)
setGlobalx(floor(rand(1) *size(train,2))+1);

display('training KNN on subspace learned by built-in pca'); %DONE
tic;
  [U_k, bestKPCA, bestNPCA, testAccuracySubspacePCA, ~] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @pca);
timepca = toc;
display('done');

display('training KNN on subspace learned by stochastic power method'); %DONE
tic;
 [U_k, bestKSPM, bestNSPM, testAccuracySubspaceSPM, varspm] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @spm);
timespm = toc;
display('done');

display('training KNN on subspace learned by incremental ipca'); %DONE
tic;
 [U_k, bestKIPCA, bestNIPCA, testAccuracySubspaceIPCA, varipca] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @ipca);
timeipca = toc;
display('done');

display('training KNN on subspace learned by MSG - may take up to 10 mins'); %DONE
tic;
 [U_k, bestKMSG, bestNMSG, testAccuracySubspaceMSG, varmsg] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @msg);
timemsg = toc;
display('done');

display('training KNN on subspace learned by SVD PCA'); 
tic;
 [U_k, bestKSVDPCA, bestNSVDPCA, testAccuracySubspaceSVDPCA, varsvdpca] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @svdpca);
timesvdpca = toc;
 display('done');

display('training KNN on a SPARSE subspace learned by SPCA - may take up to 5 mins');
tic; 
[U_k, bestKSPCA, bestNSPCA, testAccuracySubspaceSPCA, varspca] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @spca);
timespca = toc;
display('done');


%% train Tree-bagged model with subspaces learned by above algorithms
% display('training bagged tree on subspace');
% mdlLearned2 = TreeBagger(20,train',trainlabels);
%testing:
%[predictedLabelsTest2, ~] = predict(mdlLearned2,test');
%predictedLabelsTest2 = cellfun(@str2num, predictedLabelsTest2);
%testAccuracySubspace2 = sum(predictedLabelsTest2 == testlabels')/length(predictedLabelsTest2);


%% train and test model on full-dimensional training data for comparison. 
%MODEL1: KNN: Only hyperparameter is numneighbors
display('training KNN with ALL dimensions');
mdlAll1 = fitcknn(train', trainlabels, 'NumNeighbors', 1);
%testing:
[predictedLabelsTest1, ~] = predict(mdlAll1, test');
testAccuracyAll1 = sum(predictedLabelsTest1 == testlabels')/length(predictedLabelsTest1);

%MODEL2: bagged trees: numtrees (default 500)
% display('training bagged tree');
% mdlAll2 = TreeBagger(20,train',trainlabels);
%testing:
% [predictedLabelsTest2, ~] = predict(mdlAll2,test');
% predictedLabelsTest2 = cellfun(@str2num, predictedLabelsTest2);
% testAccuracyAll2 = sum(predictedLabelsTest2 == testlabels')/length(predictedLabelsTest2);


%% output results

fprintf('accuracy of %d-dim subspace on %d-NN learned by PCA: %f in time %f with captured variance %d\n',...
    bestKPCA, bestNPCA, testAccuracySubspacePCA, timepca, dataVariance);
fprintf('accuracy of %d-dim subspace on %d-NN learned by SPM: %f in time %f with captured variance %d\n',...
    bestKSPM, bestNSPM, testAccuracySubspaceSPM, timespm, varspm);
fprintf('accuracy of %d-dim subspace on %d-NN learned by IPCA: %f in time %f with captured variance %d\n',...
    bestKIPCA, bestNIPCA, testAccuracySubspaceIPCA, timeipca, varipca);
fprintf('accuracy of %d-dim subspace on %d-NN learned by MSG: %f in time %f with captured variance %d\n',...
    bestKMSG, bestNMSG, testAccuracySubspaceMSG, timemsg, varmsg);
fprintf('accuracy of %d-dim subspace on %d-NN learned by SVD PCA: %f in time %f with captured variance %d\n',...
    bestKSVDPCA, bestNSVDPCA, testAccuracySubspaceSVDPCA, timesvdpca, varsvdpca );
fprintf('accuracy of %d by %d-dim subspace on %d-NN learned by SPCA: %f in time %f with captured variance %d\n',...
    maxDimension, bestKSPCA,  bestNSPCA, testAccuracySubspaceSPCA, timespca, varspca);

% fprintf('accuracy of %d-dim subspace on BaggedTree: %d\n', bestK, testAccuracySubspace2);
fprintf('accuracy of entire data on KNN: %f\n', testAccuracyAll1);
% fprintf('accuracy of entire data on BaggedTree: %d\n', testAccuracyAll2);


end


%% train model on learned k-dimensional subspace, determine k via cross-val
function [U, bestK, bestN, chngdir, var, vararray] = crossVal(train, trainlabels, dev, devlabels, maxK, fcnHandle)
    devAc     = [];
    bestK     = 0;
    bestAcc   = 0;
    N         = [1]; %[1, 2, 4, 8, 16];  %JUST USE K = 1 BC THAT'S BEST
    maxN      = 10;
    bestN     = 3;   %actually tune this hyperparameter as well
    neighbors = 1;   %neighbors = 1:maxN:
    d         = size(train', 2);
    var       = 0;
    vararray  = [];
    
    chngdir = 0;
    %how to handle each algorithm, some require special attention...
    if (isequal(fcnHandle, @pca))
       U = fcnHandle(train'); 
    elseif (isequal(fcnHandle, @spca))
        [B SV L D PATHS] = spca(train', [], maxK, inf, -1200);
        U = B;
    else
       U = fcnHandle(train', maxK); %learn the full uncorrelated subspace via algo
    end
    
    if (size(U, 1) ~= size(train, 1))
        size(train)
        size(U)
        error('U not properly sized');
    end
    
    %begin hyperparameter tuning
    for neighbors = 1:length(N);
        for k = 1:maxK 
            top_k = U(:, 1:k); %first two columns are top two principle components
            C = top_k'*train;  %k by n
            devTest = top_k'*dev;

            if (mod(k, 25) == 0) 
                 fprintf('----training KNN with k= %d\n', k); 
            end
            mdl1 = fitcknn(C', trainlabels, 'NumNeighbors', N(neighbors));
            [predictedDevlabels, ~] = predict(mdl1, devTest');

            acc = sum(predictedDevlabels == devlabels')/length(devlabels);
            devAc = [devAc acc];
            if (acc > bestAcc) 
                bestAcc = acc;
                bestK = k;
                bestN = N(neighbors);
            end
        end
    end
    %plot graphs of accuracy vs dimension of learned subspace
    fig = figure;
    plot(devAc); hold on;
    xlabel('Number of Principle Components');
    ylabel('Accuracy');
    if (isequal(fcnHandle, @pca))
       title('Accuracy of KNN trained on subspace learned by PCA'); 
       hold off;
       print(fig,'cross-val-PCA','-dpng');
       
       mkdir('PCA-Faces');
       cd('PCA-Faces');
       chngdir = 1;
    elseif (isequal(fcnHandle, @spm))
        title('Accuracy of KNN trained on subspace learned by SPM');
        hold off;
        print(fig,'cross-val-SPM','-dpng');
        
        %plot variance captured by each principle component
        [var, vararray] = calcVariance(U, train);
        f = figure;
        plot(vararray); hold on;
        title('Variance Captured by First X Principle Components, SPM');
        hold off;
        print(f, 'Var SPM', '-dpng');
   
        mkdir('SPM-Faces');
        cd('SPM-Faces');
        chngdir = 1;
    elseif (isequal(fcnHandle, @ipca))
        title('Accuracy of KNN trained on subspace learned by IPCA'); 
        hold off;
        print(fig,'cross-val-IPCA','-dpng');
        
        %plot variance captured by each principle component
        [var, vararray] = calcVariance(U, train);
        f = figure;
        plot(vararray); hold on;
        title('Variance Captured by First X Principle Components, IPCA');
        hold off;
        print(f, 'Var IPCA', '-dpng');
           
        mkdir('IPCA-Faces');
        cd('IPCA-Faces');
        chngdir = 1;
    elseif (isequal(fcnHandle, @msg))
        title('Accuracy of KNN trained on subspace learned by MSG');
        hold off;
        print(fig,'cross-val-MSG','-dpng');
        
        %plot variance captured by each principle component
        [var, vararray] = calcVariance(U, train);
        f = figure;
        plot(vararray); hold on;
        title('Variance Captured by Each Principle Component, MSG');
        hold off;
        print(f, 'Var MSG', '-dpng');
        
        mkdir('MSG-Faces');
        cd('MSG-Faces');
        chngdir = 1;
    elseif (isequal(fcnHandle, @spca))
        title('Accuracy of KNN trained on subspace learned by SPCA');
        hold off;
        print(fig,'cross-val-SPCA','-dpng');
        
        %plot variance captured by each principle component
        [var, vararray] = calcVariance(U, train);      
        f = figure;
        plot(vararray); hold on;
        title('Variance Captured by First X Principle Components, SPCA');
        hold off;
        print(f, 'Var SPCA', '-dpng');
        
        mkdir('SPCA-Faces');
        cd('SPCA-Faces');
        chngdir = 1;
        
    elseif (isequal(fcnHandle, @svdpca))
        title('Accuracy of KNN trained on subspace learned by SVDPCA');
        hold off;
        print(fig,'cross-val-SVDPCA','-dpng'); 
        
        %plot variance captured by each principle component
        [var, vararray] = calcVariance(U, train);
        f = figure;
        plot(vararray); hold on;
        title('Variance Captured by First X Principle Components, SVDPCA');
        hold off;
        print(f, 'Var SVDPCA', '-dpng');
        
        mkdir('SVDPCA-Faces');
        cd('SVDPCA-Faces');
        chngdir = 1;
        
    end        
    U = U(:, 1:bestK);
    
end

%% center and normalize
function [X, mean] = candN(X)
    mean = sum(X, 2)/size(X, 2);
    stdtrain = std(X');
    Xcenter = bsxfun(@minus, X, mean);
    X = bsxfun(@rdivide, Xcenter, stdtrain');
end

function [U_k, bestK, bestN, testAccuracy, var, vararray] = ...
trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, fcnHandle)
    var = 0;
    vararray = [];
    [U_k, bestK, bestN, chngdir, var, vararray] = crossVal(train, trainlabels, dev, devlabels, maxK, fcnHandle);
    
    randface = getGlobalx();
    sample = train(: ,randface);
    reconstruct(sample, U_k);
    
    t = figure;
    tt = figure;
    for i = 1:5
       [im, im2] = eigenface(U_k(:, i)); 
       figure(t);
       imshow(im);
       str = sprintf(' %d th principle eigenface', i);
       title(str);
       str = sprintf('eigenface_number_%d', i);
       print(t, str,'-dpng');
       
       figure(tt);
       imshow(im2);
       str = sprintf(' %d th thresholded eigenface', i);
       title(str);
       str = sprintf('thresholded_eigenface_number_%d', i);
       print(tt, str,'-dpng');
    end

    if chngdir 
        cd('../');
    end
    
    mdlLearned1 = fitcknn((U_k'*train)', trainlabels, 'NumNeighbors', bestN);
    [predictedLabelsTest1, ~] = predict(mdlLearned1,(U_k'*test)');
    testAccuracy = sum(predictedLabelsTest1 == testlabels')/length(predictedLabelsTest1);

end

function setGlobalx(val)
    global x
    x = val;
end


function r = getGlobalx
    global x
    r = x;
end

%reconstruct a face using different numbers of the principle 
%components from the learned subspace
function [reconstructed_face] = reconstruct(sample, U_k) 

    
    original = reshape(sample, 192, 168);
    for k = 1:6 
        ff = figure;
        reconstructed_face = U_k(:,1:k*5)*U_k(:,1:k*5)'*sample;
        recon = reshape(reconstructed_face, 192, 168);
        imshow(recon);
        str = sprintf('face sample reconstructed with %d PCs', k*5); 
        title(str);
        print(ff, str, '-dpng');
    end
    
    fff = figure;
    imshow(original);
    str = sprintf('original face');

end

function [im, im2] = eigenface(U_k)

    s = U_k; 
    s = (s + abs(min(s)))/(max(s)); %rescale
    im = reshape(s, 192, 168);
    
    
    %only keep components of the principle eigenvector with most weight
    throwout = find(s < (mean(s) + std(s)));
    s(throwout) = 0;
    im2 = reshape(s, 192, 168); 
end

function [v, array] = calcVariance(U, X)
    v = 0;
    array = [];
    if (isempty(U))
        v = trace(X'*X);
        array = zeros(size(X, 2), 1);
    else
        v = trace(U'*X*X'*U);
        for j = 1:size(U, 2) %for each principles component
           array = [array trace((U(:, 1:j)'*X)*(X'*U(:, 1:j)))]; 
        end
    end
end