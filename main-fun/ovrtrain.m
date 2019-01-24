function [model] = ovrtrain(train_labels, train_data, cmd)
% one vs all LIBSVM
% addpath .\svm-mat-2.89-3
labelSet = unique(train_labels);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);
train_data = double(train_data);
for i=1:labelSetSize
    models{i} = svmtrain(double(train_labels == labelSet(i)), train_data, cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);

clear train_data;clear train_labels;
end