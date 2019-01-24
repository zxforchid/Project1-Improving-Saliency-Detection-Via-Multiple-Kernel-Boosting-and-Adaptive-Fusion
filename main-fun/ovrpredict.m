function [pred, ac, decv] = ovrpredict(test_labels, test_data, model)
% one vs all LIBSVM
labelSet = model.labelSet;
labelSetSize = length(labelSet);
models = model.models;
decv= zeros(size(test_labels, 1), labelSetSize);
test_data = double(test_data);
for i=1:labelSetSize
  [l,a,d] = svmpredict(double(test_labels == labelSet(i)), test_data, models{i});
   decv(:, i) = d * (2 * models{i}.Label(1) - 1);
end

[tmp,pred] = max(decv, [], 2);
pred = labelSet(pred);
ac = sum(test_labels==pred) / size(test_data, 1);

clear test_data;clear test_labels;
end
