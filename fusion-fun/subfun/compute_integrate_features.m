function result = compute_integrate_features(saliencymap,img)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% 计算显著图融合所需的各种特征
% reference《comparing salient object detection results without ground truth》
% copyright by xiaofei zhou,shanghai university,shanghai,china
% current version: 07/09/2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt = [0:1/11:1];
tt = tt(2:end-1);

FC=0;FH=0;FCS=0;FNC=0;FB=0;

FC = saliency_coverage(saliencymap,tt);
FCP = saliency_map_compactness(saliencymap);% with something wrong
FH = saliency_histogram(saliencymap);
FCS = color_separation(saliencymap,img);
FNC = segmentation_quality(saliencymap,img,[0.5,0.75,0.95]);% too slow
FB = boundary_quality(saliencymap,img);

result = [FC,FCP,FH,FCS,FNC,FB];

clear saliencymap img FC FH FCS FNC FB tt

end