%% scaleForSVM_test
clear;
clc;
%%
load wine;

train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];

test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];

%%
[train_scale,test_scale] = scaleForSVM(train_wine,test_wine,0,1);

%%
[train_wine,pstrain] = mapminmax(train_wine');
pstrain.ymin = 0;
pstrain.ymax = 1;
[train_wine,pstrain] = mapminmax(train_wine,pstrain);

[test_wine,pstest] = mapminmax(test_wine');
pstest.ymin = 0;
pstest.ymax = 1;
[test_wine,pstest] = mapminmax(test_wine,pstest);

train_wine = train_wine';
test_wine = test_wine';