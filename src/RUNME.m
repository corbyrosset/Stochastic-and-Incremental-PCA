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
%WARNING: do not normalize or mean center data because data is grayscale 
%image and it makes no sense for the mean to be 0 (meaning negative pixels
%exist!...

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
global maxDimension; %VERY IMPORTANT, but only for sparse PCA...
maxDimension = 1200; %why this can't be on the above line is beyond me...
maxK      = 80; %VERY IMPORTANT must be significantly less than dimensionality
%because some algorithms run in O(d*k^2) PER EXAMPLE



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
train = (1/255)*train;
dev = (1/255)*dev;
test = (1/255)*test;
display('training KNN on subspace learned by built-in pca'); %DONE
% [U_k, bestKPCA, bestNPCA, testAccuracySubspacePCA] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @pca);
display('done');

display('training KNN on subspace learned by stochastic power method'); %DONE
%  [U_k, bestKSPM, bestNSPM, testAccuracySubspaceSPM] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @spm);
display('done');

display('training KNN on subspace learned by incremental pca'); %DONE
%[U_k, bestKIPCA, bestNIPCA, testAccuracySubspaceIPCA] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @ipca);
display('done');

display('training KNN on subspace learned by MSG'); %DONE
 [U_k, bestKMSG, bestNMSG, testAccuracySubspaceMSG] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @msg);
display('done');

display('training KNN on a SPARSE subspace learned by SPCA - may take up to 5 mins');
% [U_k, bestKSPCA, bestNSPCA, testAccuracySubspaceSPCA] = trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, @spca);
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

fprintf('accuracy of %d-dim subspace on %d-NN learned by PCA: %f\n', bestKPCA, bestNPCA, testAccuracySubspacePCA);
fprintf('accuracy of %d-dim subspace on %d-NN learned by SPM: %f\n', bestKSPM, bestNSPM, testAccuracySubspaceSPM);
fprintf('accuracy of %d-dim subspace on %d-NN learned by IPCA: %f\n', bestKIPCA, bestNIPCA, testAccuracySubspaceIPCA);
fprintf('accuracy of %d-dim subspace on %d-NN learned by MSG: %f\n', bestKMSG, bestNMSG, testAccuracySubspaceMSG);
fprintf('accuracy of %d by %d-dim subspace on %d-NN learned by SPCA: %f\n', maxDimension, bestKSPCA,  bestNSPCA, testAccuracySubspaceSPCA);

% fprintf('accuracy of %d-dim subspace on BaggedTree: %d\n', bestK, testAccuracySubspace2);
fprintf('accuracy of entire data on KNN: %f\n', testAccuracyAll1);
% fprintf('accuracy of entire data on BaggedTree: %d\n', testAccuracyAll2);


end


%% train model on learned k-dimensional subspace, determine k via cross-val
function [U, bestK, bestN] = crossVal(train, trainlabels, dev, devlabels, maxK, fcnHandle)
    devAc     = [];
    bestK     = 0;
    bestAcc   = 0;
    N         = [1]; %[1, 2, 4, 8, 16];  %JUST USE K = 1 BC THAT'S BEST
    maxN      = 10;
    bestN     = 3;   %actually tune this hyperparameter as well
    neighbors = 1;   %neighbors = 1:maxN:
    d         = size(train', 2);
    
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
    elseif (isequal(fcnHandle, @spm))
        title('Accuracy of KNN trained on subspace learned by SPM');
        hold off;
        print(fig,'cross-val-SPM','-dpng');

    elseif (isequal(fcnHandle, @ipca))
        title('Accuracy of KNN trained on subspace learned by IPCA'); 
        hold off;
        print(fig,'cross-val-IPCA','-dpng');
    elseif (isequal(fcnHandle, @msg))
        title('Accuracy of KNN trained on subspace learned by MSG');
        hold off;
        print(fig,'cross-val-MSG','-dpng');
    end        
    U = U(:, 1:bestK);
    
end

%% center and normalize
function X = candN(X)
    mean = sum(X, 2)/size(X, 2);
    stdtrain = std(X');
    Xcenter = bsxfun(@minus, X, mean);
    X = bsxfun(@rdivide, Xcenter, stdtrain');
end

function [U_k, bestK, bestN, testAccuracy] = ...
    trainAndTestKNN(train, trainlabels, dev, devlabels, test, testlabels, maxK, fcnHandle)

    [U_k, bestK, bestN] = crossVal(train, trainlabels, dev, devlabels, maxK, fcnHandle);
    mdlLearned1 = fitcknn((U_k'*train)', trainlabels, 'NumNeighbors', bestN);
    [predictedLabelsTest1, ~] = predict(mdlLearned1,(U_k'*test)');
    testAccuracy = sum(predictedLabelsTest1 == testlabels')/length(predictedLabelsTest1);

end



