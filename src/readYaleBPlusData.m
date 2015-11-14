function [train, trainlabels, test, testlabels, dev, devlabels] = readYaleBPlusData(directory, devpercent, testpercent, trainpercent)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%output: X a d by n matrix, d the dimension of each image flattened,
% n the number of images
train = [];
test = [];
trainlabels = [];
dev = [];
devlabels = [];
testlabels = [];

temp = [];
%get list of files in data directory
filesStruct = dir(directory);
x = 4;
if ispc
    x = 3;
end

for i = x:length(filesStruct) %first three files are "., .., and .DS_Store"
    filesStruct(i).name;
    subDir = dir(strcat(directory,'/',filesStruct(i).name)); %construct filename
    currentlabel = str2num(filesStruct(i).name(6:7));
    for j = x:length(subDir)
        file = strcat(directory,'/',filesStruct(i).name,'/',subDir(j).name);
        if ~isempty(strfind(file, '.pgm')) &isempty(strfind(file, 'Ambient'))
            y = rand(1);
            y = y*100;
            if y < devpercent 
                dev = [dev processExample(file)];
                devlabels = [devlabels currentlabel];
            elseif y < (testpercent + devpercent)
                test = [test processExample(file)];
                testlabels = [testlabels currentlabel];
            else 
                train = [train processExample(file)];
                trainlabels = [trainlabels currentlabel];
            end
        end
    end
end
%construct class labels, one for each subject:
fprintf('loaded data of size %d by %d\n', ...
    size(train, 1), size(train, 2) + size(test, 2) + size(dev,2));

fprintf('%d train, %d dev, %d test\n', ... 
    size(train,2), size(dev,2), size(test,2));
end

function b = processExample(filename)
    example = imread(filename, 'pgm');
    %example = temp.cdata;
    %collapse into column vector
    b = reshape(example, [size(example, 1)*size(example, 2), 1]);
end

