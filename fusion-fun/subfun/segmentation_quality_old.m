function FNC = segmentation_quality_old(saliencymap,image,threshold)
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
% version3: 09/14/2015 9:37AM

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

[ms,ns] = size(saliencymap);

% %% computer Wij  由于电脑本身原因，超出计算范围，故而舍弃此种方法
% WIJ = sparse(zeros(ms*ns,ms*ns));
% % temp_img = reshape(image,ms*ns,1,3);
% % temp_img1 = repmat(temp_img,[1,ms*ns]);
% tic
% for i=1:ms*ns
%     i1 = mod(i,ms); % 行
%     j1 = (i-i1)/ms + 1; % 列
%     a1 = reshape(image(i1,j1),1,3);
%     
%     for j=1:ms*ns
%         i2 = mod(j,ms);
%         j2 = (j-i2)/ms + 1;
%         a2 = reshape(image(i2,j2),1,3);
%         
%         WIJ(i,j) = sqrt((a1-a2)*(a1-a2)');
%         
%     end
% end
% toc
% edges2 = edges2 .* (edges2 > threshold);

%% 计算前景背景segmentation的度量FNC
for is=1:num_nc
    SB = saliencymap>threshold(is);% bianary opration
    [STx,STy] = find(SB==1);% SALIENT REGION
    [BTx,BTy] = find(SB==0);
    ST = [STx,STy];% salient region coordinates
    BT = [BTx,BTy];
    [mst,~] = size(ST);
    [mbt,~] = size(BT);
    DD1 = 0;DD2 = 0;
    
    %% 计算公式(5)第一项 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W1 = [];%分母1
    W11 = [];%分子1
    for ist=1:mst
        % 中心点像素坐标及对应颜色值
        xst = ST(ist,1);
        yst = ST(ist,2);
        
        % 提出四周边界，对结果不会产生影响
        if xst>=2 && xst<=ms-1 && yst>=2 && yst<=ns-1
            
        Ist = reshape(image(xst,yst,:),1,3);
        
        % 计算分母1,i的8邻域 (计算i的所有邻域点)
        nei = [xst,yst-1;xst-1,yst-1;xst-1,yst;xst-1,yst+1;xst,yst+1;xst+1,yst+1;xst+1,yst;xst+1,yst-1];
        Ist_nei = zeros(8,3);
        dist = 0;u=0;wi=0;
        Ist_nei(1,:) = reshape(image(xst,yst-1,:),1,3);
        Ist_nei(2,:) = reshape(image(xst-1,yst-1,:),1,3);
        Ist_nei(3,:) = reshape(image(xst-1,yst,:),1,3);
        Ist_nei(4,:) = reshape(image(xst-1,yst+1,:),1,3);
        Ist_nei(5,:) = reshape(image(xst,yst+1,:),1,3);
        Ist_nei(6,:) = reshape(image(xst+1,yst+1,:),1,3);
        Ist_nei(7,:) = reshape(image(xst+1,yst,:),1,3);
        Ist_nei(8,:) = reshape(image(xst+1,yst-1,:),1,3);
        dist = sum(abs(repmat(Ist,[8,1]) - Ist_nei).^2,2);% 距离平方
        u = sum(sqrt(dist))/9;
        wi = sum(exp(-0.5*sqrt(dist)/(u+eps)));
        W1 = [W1;wi];
        
        clear Ist_nei u wi dist
        
%         % 计算分子1(排除邻域中的非背景点)
%         Ist_nei1 = [];num=0;
%         u1=0;wi1=0;dist1=0;
%         for ii=1:8
%             index = nei(ii,:);
%             [indexlabel,~] = find(index(1)==BTx & index(2)==BTy);
%             if numel(indexlabel)~=0
%                 num = num+1;
%                Ist_nei1 = [Ist_nei1;reshape(image(index(1),index(2),:),1,3);];
%             end          
%         end
%         if num~=0
%             dist1 = sum(abs(repmat(Ist,[num,1]) - Ist_nei1).^2,2);% 距离平方
%             u1 = sum(sqrt(dist1))/num;
%             wi1 = sum(exp(-0.5*sqrt(dist1)/(u1+eps)));           
%         else
%             wi1 = 0;
%         end
%         
%         W11 = [W11;wi1];
%         
%         clear Ist_nei1 u1 wi1 dist1 num
        
        end
    end
%     DD1 = sum(W11)/max(sum(W1),eps);%公式(5)第一项
     DD1 = sum(W1);
    %% 计算公式(5)第二项 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W2 = [];% 分母2
    W22 = [];% 分子2
    for ibt=1:mbt
        % 中心点像素坐标
        xbt = BT(ibt,1);
        ybt = BT(ibt,2);
        if xbt>=2 && xbt<=ms-1 && ybt>=2 && ybt<=ns-1
        Ibt = reshape(image(xbt,ybt,:),1,3);
        
        % 计算分母2(计算i的所有邻域点)
        nei = [xbt,ybt-1;xbt-1,ybt-1;xbt-1,ybt;xbt-1,ybt+1;xbt,ybt+1;xbt+1,ybt+1;xbt+1,ybt;xbt+1,ybt-1];
        Ibt_nei = zeros(8,3);
        dist = 0;u=0;wi=0;
        Ibt_nei(1,:) = reshape(image(xbt,ybt-1,:),1,3);
        Ibt_nei(2,:) = reshape(image(xbt-1,ybt-1,:),1,3);
        Ibt_nei(3,:) = reshape(image(xbt-1,ybt,:),1,3);
        Ibt_nei(4,:) = reshape(image(xbt-1,ybt+1,:),1,3);
        Ibt_nei(5,:) = reshape(image(xbt,ybt+1,:),1,3);
        Ibt_nei(6,:) = reshape(image(xbt+1,ybt+1,:),1,3);
        Ibt_nei(7,:) = reshape(image(xbt+1,ybt,:),1,3);
        Ibt_nei(8,:) = reshape(image(xbt+1,ybt-1,:),1,3);
        dist = sum(abs(repmat(Ibt,[8,1]) - Ibt_nei).^2,2);% 距离平方
        u = sum(sqrt(dist))/9;
        wi = sum(exp(-0.5*sqrt(dist)/(u+eps)));
        W2 = [W2;wi];
        
        clear Ibt_nei u wi dist
        end
    end
%     W22 = W11;  
%     DD2 = sum(W22)/max(sum(W2),eps);%公式(5)第一项
    DD2 = max(sum(W2),eps);
    %% 计算每个阈值条件下的FNC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     FNC(is) = DD1 + DD2;
    FNC(is) = DD1/max(DD2,eps);
    
    clear SB ST STx STy BT BTx BTy
end

% clear variables
clear saliencymap


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % 计算分子2(排除邻域中的非背景点)
%         Ibt_nei1 = [];num=0;
%         u1=0;wi1=0;dist1=0;
%         for ii=1:8
%             index = nei(ii,:);
%             [indexlabel,~] = find(index(1)==BTx & index(2)==BTy);
%             if numel(indexlabel)~=0
%                 num = num+1;
%                Ibt_nei1 = [Ibt_nei1;reshape(image(index(1),index(2),:),1,3);];
%             end          
%         end
%         if num~=0
%             dist1 = sum(abs(repmat(Ist,[num,1]) - Ibt_nei1).^2,2);% 距离平方
%             u1 = sum(sqrt(dist1))/num;
%             wi1 = sum(exp(-0.5*sqrt(dist1)/(u1+eps)));           
%         else
%             wi1 = 0;
%         end
%         
%         W22= [W22;wi1];
%         
%         clear Ibt_nei1 u1 wi1 dist1 num