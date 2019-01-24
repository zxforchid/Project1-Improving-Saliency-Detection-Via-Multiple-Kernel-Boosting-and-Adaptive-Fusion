function fh = saliency_histogram(saliencymap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% computer saliency_histogram feature
% input:
% saliencymap 显著图
% output:
% fh      输出特征FH
% reference paper:
% <comparing salient object detection results without ground truth>
% written by xiaofei zhou,shanghai university,shanghai,china
% current version:07/08/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_fh   = 20;% 直方图的bins数目
temp = saliencymap;
clear saliencymap

% construct bins
step = 1/num_fh;
bins = 0:step:1;

% 统计直方图
temphist = zeros(1,num_fh);
for i=1:num_fh
    a1 = temp>=bins(i);
    a2 = temp<bins(i+1);
    a3 = a1.*a2;
    temphist(i) = sum(sum(a3));
end
temphist(num_fh) = temphist(num_fh) + sum(sum(((temp==bins(num_fh+1)))));% 考虑边界因素

fh = temphist / max( sum(temphist), eps );

clear temphist temp 

end
