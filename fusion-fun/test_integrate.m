% test saliency_map_compactness
% 07/08/2015
clear all;close all;clc
% 0_5_5303_gt
img = imread('.\integrate_fun\test_quality_img\0_0_272_color.jpg');
gtimg = imread('.\integrate_fun\test_quality_img\0_0_272_gt.png');
saliencymap = imread('.\integrate_fun\test_quality_img\0_0_272_ST.png');
saliencymap = double(saliencymap);
saliencymap=(saliencymap-min(saliencymap(:)))/(max(saliencymap(:))-min(saliencymap(:))+eps);
img = imresize(img,[size(saliencymap,1) size(saliencymap,2)]);
% figure,
% subplot(1,3,1),imshow(img,[]),title('color')
% subplot(1,3,2),imshow(gtimg,[]),title('gt')
% subplot(1,3,3),imshow(saliencymap,[]),title('saliency')

tt = [0:1/11:1];
tt = tt(2:end-1);

%% test_functions
FC=0;FH=0;FCS=0;FNC=0;FB=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nsaliecny coverage FC...');tic
FC = saliency_coverage(saliencymap,tt);
toc

fprintf('\nsaliencymap compactness FCP...');tic
if sum(sum(saliencymap))==0
    FCP = zeros(1,30);
else
    FCP = saliency_map_compactness(saliencymap);
end
toc

fprintf('\nsaliency histogram FH...');tic
FH = saliency_histogram(saliencymap);
toc

fprintf('\ncolorseparation FCS...');tic
FCS = color_separation(saliencymap,img);
toc

fprintf('\nboundary quality FB...');tic
FB = boundary_quality(saliencymap,img);
toc

fprintf('\nsegmentation quality FNC...');tic
FNC = segmentation_quality(saliencymap,img,[0.5,0.75,0.95]);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = [FC,FCP,FH,FCS,FNC,FB];
% result = extract_qualitiy_features(saliencymap,img);