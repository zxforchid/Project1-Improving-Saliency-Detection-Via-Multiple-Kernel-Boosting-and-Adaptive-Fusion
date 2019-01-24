function FCS = color_separation(saliencymap,image)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:07/08/2015
% version2: 09/15/2015 21:32PM
% 使得计算更精确简洁
% 
% computer color_separation feature
% input:
% saliencymap 显著图
% image       彩色图像
% output:
% result      输出特征FCS
% reference paper:
% <comparing salient object detection results without ground truth>
% written by xiaofei zhou,shanghai university,shanghai,china
% current version:09/15/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial
nh         = 16;
nhh        = 48;

IM        = cell(1,3);
IM{1,1}   = image(:,:,1);
IM{1,2}   = image(:,:,2);
IM{1,3}   = image(:,:,3);

% constuct RGB hist bins
step = 255/nh;
bins = 0:step:255;

% compute HS and HG
hs = []; % salient region
hg = []; % background region
sumsalient_s = sum(sum(saliencymap));
backgroundmap = 1-saliencymap;
sumsalient_g = sum(sum(backgroundmap));

for i=1:3
    % 每个channel进行加权直方图统计 
    temp_s = zeros(1,nh);
    temp_g = zeros(1,nh);
    for j=1:nh
        a1 = IM{1,i}>=bins(j);
        a2 = IM{1,i}<bins(j+1);
        a3 = a1.*a2;
        temp_s(j) = sum(sum(saliencymap.*a3));
        temp_g(j) = sum(sum(backgroundmap.*a3));        
    end       
    
    % 考虑终点边界
    a4 = IM{1,i}==bins(nh+1);
    temp_s(nh) = temp_s(nh) + sum(sum(saliencymap.*a4));
    temp_g(nh) = temp_g(nh) + sum(sum(backgroundmap.*a4));
    
    % 每个channel归一化一次    
    hs = [hs,temp_s/max(sumsalient_s,eps)];        
    hg = [hg,temp_g/max(sumsalient_g,eps)];
    
end

% compute FCS
HH=[hs;hg];
FCS = sum(min(HH))/nhh;

% clear variables
clear saliencymap hs hg HH temp_s temp_g
end