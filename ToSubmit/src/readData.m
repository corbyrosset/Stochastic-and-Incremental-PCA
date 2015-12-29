function [train, trainlabels, test, testlabels] = readData(directory)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%output: X a d by n matrix, d the dimension of each image flattened,
% n the number of images
train = [];
test = [];
temp = [];
%get list of files in data directory
filesStruct = dir(directory);
x = 4;
if ispc
    x = 3;
end
for i = x:length(filesStruct) %first three files are "., .., and .DS_Store"
    file = [directory filesStruct(i).name]; %construct filename
    if (strfind(file, 'normal')); %normal is going to be in test set
        test = [test processExample(file)];
    else
        train = [train processExample(file)]; %append to data matrix
    end
end
%construct class labels, one for each subject:
trainlabels = ones(9, 1); %first subject only has 10 pictures
testlabels = [1:1:15];
for i = 2:15
    %subsequent subjects have 11 pictures
    trainlabels = [trainlabels; i*ones(10, 1)];
end
train = double(train);
test = double(test);
fprintf('loaded data of size %d by %d\n', ...
    size(train, 1), size(train, 2) + size(test, 2));
end

function b = processExample(filename)
    temp = importdata(filename);
    example = temp.cdata;
    %collapse into column vector
    b = reshape(example, [size(example, 1)*size(example, 2), 1]);
end

