function FB = boundary_quality(saliencymap,image)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1: 07/08/2015
% scale 用于图像的缩放，不是edges-master中的参数
% computer boundary_quality feature
% version2: 09/15/2015  21:33PM
% 舍去边界像素点 提高FB的值
% input:
% saliencymap 显著图
% image       彩色图像
% output:
% result      输出特征FB  4维的
% reference paper:
% <comparing salient object detection results without ground truth>
% written by xiaofei zhou,shanghai university,shanghai,china
% current version: 09/15/2015 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial 
SCALES = [0.25,0.5,0.75,1.0];
[mt,nt] = size(SCALES);
num_fb = mt*nt;
FB = zeros(1,num_fb);

%% compute saliency boundary map(后需优化可以尝试用矩阵操作)
for ii=1:num_fb
    % 多尺度之单尺度
scale = SCALES(ii);
temp_sal = imresize(saliencymap,scale);
temp_sal = normalize(temp_sal);
[ms,ns] = size(temp_sal);

temp_img = imresize(image,scale);
% temp_img = normalize(temp_img);

%% compute EI edge map
model=edgesDetect_model(scale);
EI = edgesDetect(temp_img,model);
clear model

%% compute BM  boundary map
BM = zeros(ms-2,ns-2);
wps = zeros(ms-2,ns-2);
for i=2:ms-1
    for j=2:ns-1       
        mp = temp_sal(i,j);% p点
        mp12 = 0;maxdiff = 0;wp = 0;
        
        % compute edge neighbor 与边缘线垂直 p1p2
        % 计算边缘线强度
        temp = zeros(4,1);
        temp(1,1) = abs(EI(i,j-1)-EI(i,j+1));       
        temp(2,1) = abs(EI(i-1,j-1)-EI(i+1,j+1));
        temp(3,1) = abs(EI(i-1,j)-EI(i+1,j));
        temp(4,1) = abs(EI(i-1,j+1)-EI(i+1,j-1));
        
        % find the edge direction and P1 P2
        % 确定边缘线方向 进而确定P1 P2的位置
        [~,maxindex] = max(temp);
        maxindex = maxindex(end);
        switch maxindex
            case 1
                mp1 = temp_sal(i,j-1);
                mp2 = temp_sal(i,j+1);
                
            case 2
                mp1 = temp_sal(i-1,j-1);
                mp2 = temp_sal(i+1,j+1);
                
            case 3
                mp1 = temp_sal(i-1,j);
                mp2 = temp_sal(i+1,j);
                
            case 4
                mp1 = temp_sal(i-1,j+1);
                mp2 = temp_sal(i+1,j-1);
        end
        % 对应于 |M(p1) - M(p2)|
        mp12 = abs(mp1-mp2);
        
        % 对应于 max(|M(p) - M(p)|,|M(p) - M(p2)|)
        maxdiff = max([abs(mp-mp1),abs(mp-mp2)]);
        
        % 对应于Wp
        wp = mp*maxdiff;     
        
        % 对应于BM(P)
        BM(i-1,j-1) = wp*mp12;        
        
        % 收集wp 用于归一化
        wps(i-1,j-1) = wp;
        
    end
end
% 对应于公式 7
sumwps = sum(sum(wps));
BM = BM/max(sumwps,eps);

% % 填充边界 对折填充
% BM(2:ms-1,1) = BM(2:ms-1,3);
% BM(2:ms-1,ns) = BM(2:ms-1,ns-2);
% BM(1,:) = BM(3,:);
% BM(ms,:) = BM(ms-2,:);

% 去除 EI的边界点 使之与BM相同尺寸
EI1 = EI(2:ms-1,2:ns-1);
EI1 = normalize(EI1);

%% compute fb feature
FB(ii) = sum(sum(BM.*EI1));


%% clear 
clear BM EI EI1
end

end

