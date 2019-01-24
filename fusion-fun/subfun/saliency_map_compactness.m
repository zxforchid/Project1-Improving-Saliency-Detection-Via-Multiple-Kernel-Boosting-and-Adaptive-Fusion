function FCP = saliency_map_compactness(saliencymap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% computer saliency_map_compactness feature
%
% Version2:
% 实行跳点检测,缩小搜索范围，加快搜索进度
% 
% version3:08/04/2015
% 均是continue
% delta=1
% 
% version4: 09/13/2015  16:03PM
% 加快计算速度，提升准确率
% step1=2;step2=2;delta = 8;
% input:
% saliencymap 显著图
% output:
% FCP      输出特征FCP 1*(3*10)
% reference paper:
% <comparing salient object detection results without ground truth>
% written by xiaofei zhou,shanghai university,shanghai,china
% current version: 09/13/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial 
pvalues = [0.25,0.5,0.75]; % portion
FCP = [];
tt = [0:1/11:1];
tt = tt(2:end-1);

step1=2;step2=2;

temp_sal = imresize(saliencymap,[100,100]);
temp_sal=(temp_sal-min(temp_sal(:)))/(max(temp_sal(:))-min(temp_sal(:))+eps);
SA = sum(sum(temp_sal));% sum of saliency map

%% integral image
% compute integral image of saliencymap MI = Mt+1   NI = Nt+1
defaultoptions=struct('ScaleUpdate',1/1.2,'Resize',false,'Verbose',true);
IntergralImages = GetIntergralImages(temp_sal,defaultoptions);
IP = IntergralImages.ii;
[mI,nI] = size(IP);
clear IntergralImages saliencymap

%% search WP and compute FCP
for ss=1:length(pvalues) % 3 scales
    tmp_SA = pvalues(ss)*SA;
    Sdelta_area = [];
    Sdelta_location = [];
%%%%%%%%%%%%%%%%%%%%%%%%%% SINGLE SCALE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
delta = 15;
%---------- 一次搜索 ----------%
for i1=2:step1:mI-1
    for j1=2:step1:nI-1
        S1 = IP(i1,j1);
        
        % ------------ 二次搜索 ----------%
        for i2=i1:step2:mI
            for j2=j1:step2:nI
                S2 = IP(i2,j2);
                S3 = IP(i1,j2);
                S4 = IP(i2,j1);
                
                Sdelta = S2 + S1 - S3 - S4;
                
                %--- 加快搜索进度 ----% 0.95~1.05   
                if Sdelta<=0.95*tmp_SA
                    continue;
                end
                
                if Sdelta>=1.05*tmp_SA       
                    break;
                end
                
                 % -----------------进一步缩小WP范围-------------------------%
                if Sdelta>=tmp_SA-delta && Sdelta<=tmp_SA+delta
%                 if (Sdelta>0.8*tmp_SA) && (Sdelta<1.2*tmp_SA) 
%                     AA = [AA;Sdelta,i1-1,j1-1,i2-1,j2-1;];
%                     [AAminvalue,AAminindex]=min(abs(AA(:,1) - tmp_SA));
%                     LL = AA();
                    Sdelta_area = [Sdelta_area;(i2-i1+1)*(j2-j1+1)];
                    Sdelta_location = [Sdelta_location;i1-1,j1-1,i2-1,j2-1;];
                    
                end
                % ----------------------------------------------------------%
            end
        end
        % ------------ 二次搜索 ----------%
        
    end
end
% toc
clear i1 j1 i2 j2 S1 S2 S3 S4 Sdelta tmp_SA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[minvalue,index] = min(Sdelta_area);
index = find(Sdelta_area==minvalue);
% index = index(round(length(index)/2));% 取中间的那个
index = index(end);
location = Sdelta_location(index,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 取平均不取最小
% if length(Sdelta_area)>3
%    [maxvalue,maxindex] = max(Sdelta_area);
%    Sdelta_location(maxindex,:) = [];
%    [minvalue,minindex] = min(Sdelta_area);
%    Sdelta_location(minindex,:) = [];  
% end
% location = mean(Sdelta_location);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute FCP
x1 = location(1);y1 = location(2);
x2 = location(3);y2 = location(4);
WP = temp_sal(x1:x2,y1:y2);

% % test
% Objects = [x1,y1,(x2-x1+1),(y2-y1+1)];
% ShowDetectionResult(temp_sal,Objects)
% % % %

FCP = [FCP,saliency_coverage(WP,tt)];

% over one scale and clear some variables
clear WP i1 j1 i2 j2 index location 

end

% clear all
clear Sdelta_area Sdelta_location Objects  IP

end


