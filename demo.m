% the main function
% xiaofei zhou, IVP Lab, shanghai university,shanghai,china
% 

clear all;close all;clc

%% initial
fprintf('\n initial ---------------------------------------------------\n')
colorImname = '951.jpg';
initialImgname = '951_ST.png';
saveComplementPath = '.\testdata\resultImg\';
superPath = '.\testdata\SLICimg\';
Thigh = 2.2;Tlow = 0.2;
colorImg = imread(colorImname);
salInitial = imread(initialImgname);

figure,imshow(colorImg,[]),title('original image')
figure,imshow(salInitial,[]),title('initial saliency map')

%% generate implementary saliency map
fprintf('\n generate implementary saliency map-------------------------\n')
[salComplement, salComplementFilter]  = ...
    generateComplementSal(Thigh, Tlow, colorImname,initialImgname,saveComplementPath,superPath);

figure,imshow(salComplement,[]),title('complement saliency map')


%% fusion of implementary and initial slaiency map
fprintf('\n  fusion of implementary and initial slaiency map ----------\n')
% create quality feature
complementFeature = extract_qualitiy_features(salComplement,colorImg);
initialFeature = extract_qualitiy_features(salInitial,colorImg);

testData = [complementFeature;initialFeature];
testLabel = ones(size(testData,1),1);

% load quality score model
load('.\quality-model-mat\ST_qualitymodel')

% get quality score
[prel, acc , Decv] = ovrpredict(testLabel, testData, model);
[tmpDecv,preLabel] = max(Decv, [], 2);
tmpComScore = tmpDecv(1,:);
tmpInitScore = tmpDecv(2,:);
scoreComplement = tmpComScore./max((tmpComScore + tmpInitScore),eps);
scoreInitial = tmpInitScore./max((tmpComScore + tmpInitScore),eps);

% fusion
salComplement = normalize(salComplement);
salInitial = normalize(salInitial);
salFusion = scoreComplement*salComplement + scoreInitial*salInitial;
salFusion = normalize(salFusion);
figure,imshow(salFusion,[]),title('fusion saliency map')

salFusion = 255 * salFusion;
imwrite(salFusion,['.\testdata\resultImg\',colorImname(1:end-4),'.png']);



%% over
msgbox('Well Done, BOY!!!')
