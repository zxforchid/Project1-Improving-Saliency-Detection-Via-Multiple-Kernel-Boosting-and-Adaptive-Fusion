function  quality_biased_integrate(path,tmpname,astrong,aweak,suffix,delta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% 基于质量评价的融合算法
% VERSION2:
% 3 INTEHRATION METHOD BASED ON QUALITY
% 
% VERSION3:
% 加入距离均值差delta
% reference：《comparing salient object detection results without ground truth》
% copyright by xiaofei zhou,shanghai university,shanghai,china
% current version: 07/13/2015  18:48PM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1 = 0.618*delta;
d2 = 0.382*delta;

imgRoot_weaksal = path.imgRoot_weaksal;
saldir_strong = path.saldir_strong;
saldir_final = path.saldir_final;
saldir_final0 = path.saldir_final0;
clear path
    imname_strong = [saldir_strong tmpname suffix.strong]; 
    imname_weak   = [imgRoot_weaksal tmpname suffix.weak];
    im_weak      = im2double(imread(imname_weak));
    im_strong     = im2double(imread(imname_strong));
    
%% 线性加权融合 1
        integrate_salmap1 = astrong*im_strong + aweak*im_weak;
        integrate_salmap1 = normalize(integrate_salmap1);
        imwrite(integrate_salmap1,[saldir_final tmpname '_quality_integrate1.png']);
        
%         for cc=1:9
%             c1=0.1*cc;
%             c2 = 1-c1;
%             linearmap = normalize(c1*im_weak + c2*im_strong);
%             imwrite(linearmap,[saldir_final0 tmpname '_' num2str(cc) '_linear_integrate.png']);
%         end
    
% %% 选择融合 2
%         integrate_salmap2 = (im_strong.^(astrong > aweak)).*(im_weak.^(astrong < aweak));
%         integrate_salmap2 = normalize(integrate_salmap2);    
%         imwrite(integrate_salmap2,[saldir_final tmpname '_quality_integrate2.png']); 
% 
% %% 融合 3
%       if abs(astrong - aweak)<d1 && abs(astrong - aweak)>=d2
%           integrate_salmap3 = normalize(integrate_salmap2 + integrate_salmap1);
%       end
%       
%       if abs(astrong - aweak)>=d1
%           integrate_salmap3 = integrate_salmap2;
%       end
%      
%       if abs(astrong - aweak)<d2
%           integrate_salmap3 = integrate_salmap1;
%       end
%       imwrite(integrate_salmap3,[saldir_final tmpname '_quality_integrate3.png']); 
      
    clear integrate_salmap1 integrate_salmap2 integrate_salmap3 im_strong im_weak
    
end

