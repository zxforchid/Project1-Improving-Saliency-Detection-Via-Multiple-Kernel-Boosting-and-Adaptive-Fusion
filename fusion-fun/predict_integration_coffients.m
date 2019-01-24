function [predict_labels,accuracy,decvalue] = predict_integration_coffients(model,testdata,testlabel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% 利用前面训练得到的模型model,对新输入的样本进行预测
% Input:
% testdata testlabel 分别代表测试样本的特征数据及对应的样本标签
% model 代表由前述训练得到的模型
% 
% Output:
% predict_labels代表预测标签
% accuracy 表示准确度
% decvalue logistic输出的值（未经过判决函数sgn）
% 
% Current version: 07/13/2015 
% Copyright by xiaofei zhou,shanghai university,shanghai,china
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n] = size(testdata);

% model.theta = theta;
% model.acc = accuracy;
% model.lambda = lambda;
% model.iternum = Iteration;

% Add intercept term to traindata
X = [ones(m, 1) testdata];
y = testlabel;

clear testdata testlabel

theta = model.theta;
decvalue = sigmoid(X * theta);% m*(n+1) X (n+1)*1

predict_labels = predict(theta, X);

accuracy = mean(double(predict_labels == y)) * 100;

clear model X y

end