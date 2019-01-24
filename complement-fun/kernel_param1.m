function str = kernel_param1( nsvm )
%%
% version2:
% 修改核函数，利用线性核与径向基核
str = cell(nsvm,1);
pos = 1;
% cmd = ['-s 3 -p 0.5 '];% regression or classifier
cmd = ['-s 3 '];
str{pos} = [cmd '-t 0'];
% str{pos} = '-t 2';
degree = 1;
coef = 3;
% degree = 3;
% coef = 0;
for i = 1:size(degree,2)
    for j = 1:size(coef,2)
        pos = pos + 1;
        s = [cmd '-t 1 -d ' num2str(degree(i)) ' -r ' num2str(coef(j))];
        str{pos} = s;
    end
end
pos = pos + 1;
str{pos} = [cmd '-t 2'];
for i = 1:size(coef,2)
    pos = pos + 1;
    s = [cmd '-t 3 -r ' num2str(coef(i))];
    str{pos} = s;
end