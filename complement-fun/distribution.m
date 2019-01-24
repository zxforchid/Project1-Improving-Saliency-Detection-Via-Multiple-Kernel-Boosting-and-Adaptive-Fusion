function d = distribution( model )
%%
n = size(model,1);
d = cell(n,4);
for i = 1:n  
    d{i,1} = mean(model(i).SVs(model(i).sv_coef>0,:),1);
%     d{i,2} = cov(model(i).SVs(model(i).sv_coef>0,:));
    d{i,3} = mean(model(i).SVs(model(i).sv_coef<0,:),1);
%     d{i,4} = cov(model(i).SVs(model(i).sv_coef<0,:));
end
