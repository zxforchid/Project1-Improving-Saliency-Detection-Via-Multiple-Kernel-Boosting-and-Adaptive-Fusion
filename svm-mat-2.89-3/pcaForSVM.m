function [train_pca,test_pca] = pcaForSVM(train,test,threshold)
% pca pre-process for SVM
% by faruto Email:farutoliyang@gmail.com QQ:516667408
% 2009.10.27

if nargin == 2
    threshold = 90;
end

%%
[train_coef,train_score,train_latent,train_t2] = princomp(train);
%%
train_cumsum = 100*cumsum(train_latent)./sum(train_latent);
index = find(train_cumsum >= threshold);

percent_explained = 100*train_latent/sum(train_latent);
figure;
pareto(percent_explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

%% 
train_pca = train*train_coef;
train_pca = train_pca(:,1:index(1));

test_pca = test*train_coef;
test_pca = test_pca(:,1:index(1));