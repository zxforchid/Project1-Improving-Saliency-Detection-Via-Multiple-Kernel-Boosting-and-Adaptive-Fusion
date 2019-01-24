function [ alphas, model, tmodel ] = boost_SVR3( data, label, nfeature, datanum,ntype, errormethod )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version2:
% change Df(i) and beta for boost_SVR  计算错误率delta时进行改变
% △=1.5*δ，δ=sum(abs(fitvalue-truevalue))/N
% Copyright(C) by Xiaofei Zhou,shanghai university,shanghai,china
% Version3:
% 加入对错误率计算的方法的选择
% errormethod   'MSE' 'ABS'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial 
n_svm = nfeature * ntype;
tlabel = cell(n_svm,1);
tfitvalue = cell(n_svm,1); 
tmse      = cell(n_svm,1);
tscc      = cell(n_svm,1);

tbeta = []; model = []; alphas = []; tt = [];
epsilon = [];tmodel = []; mses = [];
%% distribute data and label
d1 = data.rgb; d2 = data.lab; d3 = data.hsv; d4 = data.lm; d5 = data.lbp; d6 = data.sift; d7 = data.hog; d8 = data.geodist;
l1 = label.rgb;l2 = label.lab;l3 = label.hsv;l4 = label.lm;l5 = label.lbp;l6 = label.sift;l7 = label.hog;l8 = label.geodist;

pos = 0;

clear data label
%% train all the model
mses1 = [];
str = kernel_param1( ntype );
for i = 1:nfeature
    d = eval(['d' num2str(i)]);
    l = eval(['l' num2str(i)]);
    for j = 1:ntype
        fprintf('\nfeature d%d | kernel k%d\n',i,j)
        pos = pos + 1;
        s = str{j};

        m = svmtrain(l, d, s);    
        [pred_l, mse] = svmpredict(l, d, m);
        tmodel = [tmodel; m];
        switch errormethod
            case 'MSE'
                mses = [mses;mse(2)];
                
            case 'ABS'
                mses = [mses;sum(abs(pred_l - l))/size(l,1)];
                
        end
        tfitvalue{pos} = pred_l;

        clear pred_l mse m
    end
end

%% boosting
iter = n_svm-2;
W = cell(nfeature,1);
% inital sample weight 进行特征选择，故而针对每种特征进行选择
for j = 1:nfeature
    W{j} = ones(datanum(j),1) / datanum(j);
end

% begin 
ERRORS=[];% 存储误差值
MARKS=[];% 存储误差标签
for t = 1:iter
    % one iteration, compute error
    for j = 1:n_svm
        if sum(j==tt) ~= 0 
            ERRORS = [ERRORS;+inf];
            MARKS = [MARKS,zeros(size(tfitvalue{j},1),1)];
            continue; 
        end
        % find feature index and label index(true fit value)
        fi = floor((j-1)/ntype)+1;
        l = eval(['l' num2str(fi)]);
        
        % compute deta 
        delta = 1.5*mses(j);
        
        % compute em(error rate)
        switch errormethod
            case 'MSE'
                HS_value = (tfitvalue{j}-l).*(tfitvalue{j}-l)-delta; 
                
            case 'ABS'
                HS_value = abs(tfitvalue{j}-l)-delta;
                
        end
            
        HS_mark = HS_value>0;
        error_t = sum(W{fi}.*HS_mark);
        MARKS = [MARKS,HS_mark];
        
        if error_t==0 % 排除极端情况
            ERRORS = [ERRORS;inf];            
        else
            ERRORS = [ERRORS;error_t];
        end
            
    end
    
    % find the min error
    [error_min,idx] = min(ERRORS);
    idx1 = find(ERRORS==error_min);
    idx = idx1(end);
    if error_min>0.5 
        break; 
    end
    beta_t = error_min/(1-error_min+eps);
   
    % memory this beta and model and model index
    alpha = 0.5*log(1/beta_t);
    alphas = [alphas; alpha];
    model = [model; tmodel(idx)];
    tt = [tt; idx];
    
    % updata W
    fi = floor((idx-1)/ntype)+1;% 求取对应特征号，进而获取相对应权重
    W{fi} = W{fi}.*(beta_t.^(1-MARKS(:,idx)));
%     W{fi} = W{fi}.*exp(-alpha.*mark);
    W{fi} = W{fi} / sum(W{fi});
    
    % others clear 
    ERRORS = [];
    MARKS = [];
end
alphas = [alphas tt];

clear tfitvalue mses ERRORS MARKS W 
end

