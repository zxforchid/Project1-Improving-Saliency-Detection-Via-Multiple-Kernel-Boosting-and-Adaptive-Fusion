function FNC = segmentation_quality(saliencymap,image,threshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% computer segmentation_quality feature
% input:
% saliencymap 显著图
% image       彩色图像
% threshold   前景背景阈值
% output:
% result      输出特征FCS
% 
% version2:07/08/2015
% 优化程序，加快速度
% 把FNC改为前景/背景，即分母1除以分母2
%
% version3: 09/14/2015 14:08AM
% 按照新的理解及对公式的修改，我们做出相应的改变
% 
% reference paper:
% <comparing salient object detection results without ground truth>
% written by xiaofei zhou,shanghai university,shanghai,china
% current version: 09/14/2015 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial
[mt,nt] = size(threshold);
num_nc  = mt*nt;
FNC = zeros(1,num_nc);
image = double(image);

% [ms,ns] = size(saliencymap);
% edges2 = edges2 .* (edges2 > threshold);

%% 计算前景背景segmentation的度量FNC
step = 4;
for is=1:num_nc
    
    SB = saliencymap>threshold(is);% bianary opration
    
    % 下采样，加快执行速度
    SB1 = SB(1:step:end,1:step:end);
    image1 = image(1:step:end,1:step:end,:);
    [ms,ns] = size(SB1);
    % ----- %
    
    [STx,STy] = find(SB1==1);% SALIENT REGION
    [BTx,BTy] = find(SB1==0);
    ST = [STx,STy];% salient pixels coordinates
    BT = [BTx,BTy];% background pixels coordinates
    [mst,~] = size(ST);
    [mbt,~] = size(BT);
    DD1 = 0;DD2 = 0;
    
    %% 计算公式(5)第一项 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    WMs1 = [];%分母1
    WZs1 = [];%分子1
    for ist=1:mst
%         fprintf('\n%d',ist)
        % 中心点像素坐标及对应颜色值
        xst = ST(ist,1);
        yst = ST(ist,2);
        % 提出四周边界，对结果不会产生影响 5~M-4  5~N-4  9*9的区域
        if xst>=5 && xst<=ms-4 && yst>=5 && yst<=ns-4
            [WM,WZ] = single_pixel_weight(xst,yst,BT,image1);
             WMs1 = [WMs1;WM];
             WZs1 = [WZs1;WZ];
             clear WM WZ
        end
    end
    DD1 = sum(WZs1)/max(sum(WMs1),eps);%公式(5)第一项
    
    %% 计算公式(5)第二项 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    WMs2 = [];% 分母2
    WZs2 = [];% 分子2
    for ibt=1:mbt
        % 中心点像素坐标
        xbt = BT(ibt,1);
        ybt = BT(ibt,2);
%         fprintf('\n%d',ibt)
        if xbt>=5 && xbt<=ms-4 && ybt>=5 && ybt<=ns-4
            [WM,WZ] = single_pixel_weight(xbt,ybt,ST,image1);     
             WMs2 = [WMs2;WM];
             WZs2 = [WZs2;WZ];
             clear WM WZ
        end
    end
    DD2 = sum(WZs2)/max(sum(WMs2),eps);%公式(5)第一项
    %% 计算每个阈值条件下的FNC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FNC(is) = DD1 + DD2;
    
    clear SB ST BT SB1 image1
end

% clear variables
clear saliencymap DD1 DD2 WMs1 WMs2 WZs1 WZs2


end

% 对于第一项，给定任意一点，计算分子分母对应的权重向量
function [WM,WZ] = single_pixel_weight(xsbt,ysbt,SBT,image)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WM 分母对应的权重向量
% WZ 分子对应的权重向量（提出非背景点/非前景点）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 彩色图像中心点像素值
Ist = reshape(image(xsbt,ysbt,:),1,3); 
        
%%  像素点的9*9邻域
% 分母
NI = image(xsbt-4:xsbt+4,ysbt-4:ysbt+4,:);
Ist_nei = reshape(NI,81,3);
dist = sum((repmat(Ist,[81,1]) - Ist_nei).^2,2);% 距离平方 81*1
u = sum(dist)/81;
WM = exp(-0.5*sqrt(dist)/(u+eps));% 分母 81*1 包括自身 0

% 分子 剔除非背景点/非前景点
ax = repmat((xsbt-4:xsbt+4)',1,9);
ay = repmat(ysbt-4:ysbt+4,9,1);
NI_coors = [ax(:),ay(:)];
% NI_coors1(1,1:2:161) = ax(:);
% NI_coors1(1,2:2:162) = ay(:);
% delta = repmat(NI_coors1,size(SBT,1),1) - repmat(SBT,1,length(NI_coors));
% delta1 = delta(:,1:2:end) + delta(:,2:2:end);
% delta1(delta1~=0) = 1;
% delta2 = delta1.*ones(size(delta1,1),size(delta1,2));
% delta3 = sum(delta2);
% [z1,z2] = find(delta3==size(delta1,1));

SBTX = SBT(:,1);
SBTY = SBT(:,2);
ZZ_INDEX = [];% 存储背景点/前景点
for dd=1:81
    x = NI_coors(dd,1); % 行
    y = NI_coors(dd,2); % 列
    num = 0;
    
    % 寻点
    [indexx,~] = find(x==SBTX);
    for ii=1:length(indexx)
        y1 = SBTY(indexx(ii));
        if y1==y
            num = num + 1;
        end       
    end
    
    % 存点 (不等于0表示是背景点)
    if num~=0
        ZZ_INDEX = [ZZ_INDEX;dd];
    end
    
end

% 剔除 得分子WZ
WZ = WM(ZZ_INDEX);

% OVER
clear Ist_nei NI SBT SBTX SBTY ZZ_INDEX

end
