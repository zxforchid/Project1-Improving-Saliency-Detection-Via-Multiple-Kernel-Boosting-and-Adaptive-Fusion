function [train_scale,test_scale] = scaleForSVM(train_data,test_data,ymin,ymax)
% scaleForSVM 
% by faruto Email:farutoliyang@gmail.com
% 2009.10.28

[train_data,pstrain] = mapminmax(train_data');
pstrain.ymin = ymin;
pstrain.ymax = ymax;
[train_data,pstrain] = mapminmax(train_data,pstrain);

[test_data,pstest] = mapminmax(test_data');
pstest.ymin = ymin;
pstest.ymax = ymax;
[test_data,pstest] = mapminmax(test_data,pstest);

train_scale = train_data';
test_scale = test_data';