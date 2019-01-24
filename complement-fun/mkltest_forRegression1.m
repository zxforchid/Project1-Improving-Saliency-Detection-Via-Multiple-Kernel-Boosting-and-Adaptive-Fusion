function [  conf ] = mkltest_forRegression1(beta, model, d, data,ntype)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% written for SVR
% 8种特征3种核
% 
% version2:
% 利用normal_enhanced 函数替换normalize函数，加强版的归一化
%
% Copyright(C) by xiaofei zhou,shanghai university,shanghai,china
% current version: 07/10/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(data.rgb,1);
conf  = zeros(1,n);

% features
rgb=data.rgb;
lab=data.lab;
hsv=data.hsv;
lm = data.lm;
lbp = data.lbp;
sifts = data.sift;
hogs = data.hog;
geodist = data.geodist;

l=ones(n,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 for j = 1:size(beta,1)
      idx = beta(j,2);
        m = model(j);
        switch (floor((idx-1)/ntype))
            case 0
                [pred_l, mse] = svmpredict(l, rgb, m);
                prob = distribution_prob( d, rgb, j );
                dec = pred_l;
                
            case 1
                [pred_l, mse] = svmpredict(l, lab, m);
                prob = distribution_prob( d, lab, j );
                dec = pred_l;
               
            case 2            
                [pred_l, mse] = svmpredict(l, hsv, m);
                prob = distribution_prob( d, hsv, j );
                dec = pred_l;
                
           case 3
                [pred_l, mse] = svmpredict(l, lm, m);
                prob = distribution_prob( d, lm, j );
                dec = pred_l;
                
            case 4
                [pred_l, mse] = svmpredict(l, lbp, m);
                prob = distribution_prob( d, lbp, j );
                dec = pred_l;
                
            case 5
                [pred_l, mse] = svmpredict(l, sifts, m);
                prob = distribution_prob( d, sifts, j );
                dec = pred_l;
                
            case 6
                [pred_l, mse] = svmpredict(l, hogs, m);
                prob = distribution_prob( d, hogs, j );
                dec = pred_l;
                
            case 7
                [pred_l, mse] = svmpredict(l, geodist, m);
                prob = distribution_prob( d, geodist, j );
                dec = pred_l;
        end
        conf = (conf' + beta(j,1) * dec.* prob)';
        
        % clear variables
        clear dec pred_l mse prob dec
 end
 
%% normalize
% conf=(conf-min(conf(:)))/(max(conf(:))-min(conf(:)));% max-min normalizatoin
conf = normal_enhanced(conf);

end




