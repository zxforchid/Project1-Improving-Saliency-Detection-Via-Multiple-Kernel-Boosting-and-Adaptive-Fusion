function result = extract_qualitiy_features(saliencymap,img)
% 提取质量评价特征
% 07/11/2015
% copyright by xiaofei zhou,shanghai university,shanghai,china
% 
% max-min scale [0,1]
saliencymap = double(saliencymap);
saliencymap = (saliencymap-min(saliencymap(:)))/(max(saliencymap(:))-min(saliencymap(:))+eps);
img = imresize(img,[size(saliencymap,1) size(saliencymap,2)]);
tt = [0:1/11:1];
tt = tt(2:end-1);

FC=0;FH=0;FCS=0;FNC=0;FB=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nsaliecny coverage FC...');
FC = saliency_coverage(saliencymap,tt);

fprintf('\nsaliencymap compactness FCP...');
if sum(sum(saliencymap))==0
    FCP = zeros(1,30);
else
    FCP = saliency_map_compactness(saliencymap);
end

fprintf('\nsaliency histogram FH...');
FH = saliency_histogram(saliencymap);

fprintf('\ncolorseparation FCS...');
FCS = color_separation(saliencymap,img);

fprintf('\nboundary quality FB...');
FB = boundary_quality(saliencymap,img);
% aa = pwd;
% cd(aa(1:end-27))

fprintf('\nsegmentation quality FNC...');
FNC = segmentation_quality(saliencymap,img,[0.5,0.75,0.95]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = [FC,FCP,FH,FCS,FNC,FB];

clear saliencymap img FC FH FCS FNC FB tt
end