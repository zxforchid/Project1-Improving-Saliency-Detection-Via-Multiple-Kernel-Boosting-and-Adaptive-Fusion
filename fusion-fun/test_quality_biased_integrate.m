%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% 根据前述的特征+机器方法得到的所谓的强显著性图strong,加上输入的weak,我们采用
% 图像显著性质量评价的融合算法
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@outlook.com 
% 07/13/2015  19:04PM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;close all;clc
%% image path: color image/ weak saliencymap/ strong saliencymap
imgRoot_color='.\MSRA\image\';       % color image path
imgRoot_gt='.\MSRA\gt\';       % ground truth image path
imgRoot_weaksal = '\ST\MSRA\'; % weak saliencymap
imgRoot_strongsal = '.\result_ST_MSRA\OUT-BL\strong\';% strong saliencymap
% im_feature_dir = '.\result_ST\featuremat\'; 
saldir_final='.\result_ST_MSRA\OUT-BL\final\';
% 
% if ~isdir(im_feature_dir)
% mkdir(im_feature_dir);
% end

path.imgRoot_color=imgRoot_color;
path.imgRoot_weaksal=imgRoot_weaksal;
path.imgRoot_strongsal=imgRoot_strongsal;
path.imgRoot_gt=imgRoot_gt;
path.saldir_final = saldir_final;
%% image name suffix
suffix.colorimg = '.jpg';
suffix.gt = '.png';
suffix.weak = '_SelSalPixel.png';
suffix.strong = '_strong_filter.png';

%% extract quality feature
imnames=dir([imgRoot_color '*' 'jpg']); % image nums
% if 1
% feature_strong = single(zeros(length(imnames),68));
% feature_weak = single(zeros(length(imnames),68));
load feature_strong_weak1.mat

% matlabpool local 4

% parfor ii=1:length(imnames)
% % parfor ii=2501:3000
% %     fprintf('\nthe %dst image...........\n',ii);
% % %     t = tic;
% %  [feature_strong(ii,:), feature_weak(ii,:)]= extract_feature_forintegral(path,suffix,imnames,ii);
% % %  feature_strong(ii,:) = strong;
% % %  feature_weak(ii,:) = weak;
% % 
% % %  imwrite([im_feature_dir,num2str(ii)])
% % %     timecost = toc(t);
% % %     fprintf('\nit takes time about: %.3f s\n',timecost)
% % end
% % % matlabpool close
% % save ('feature_strong_weak1.mat','feature_strong','feature_weak')
% % 
% % 
% % % matlabpool local 4
% % parfor ii=3001:3500
% %     fprintf('\nthe %dst image...........\n',ii);
% % %     t = tic;
% %  [feature_strong(ii,:), feature_weak(ii,:)]= extract_feature_forintegral(path,suffix,imnames,ii);
% % %  fea  ture_strong(ii,:) = strong;
% % %  feature_weak(ii,:) = weak;
% %  
% % %  imwrite([im_feature_dir,num2str(ii)])
% % %     timecost = toc(t);
% % %     fprintf('\nit takes time about: %.3f s\n',timecost)
% % end
% % % matlabpool close
% % 
% % save ('feature_strong_weak1.mat','feature_strong','feature_weak')
% % 
% % % matlabpool local 4
% % parfor ii=3501:4000
% %     fprintf('\nthe %dst image...........\n',ii);
% % %     t = tic;
% %  [feature_strong(ii,:), feature_weak(ii,:)]= extract_feature_forintegral(path,suffix,imnames,ii);
% % %  fea  ture_strong(ii,:) = strong;
% % %  feature_weak(ii,:) = weak;
% %  
% % %  imwrite([im_feature_dir,num2str(ii)])
% % %     timecost = toc(t);
% % %     fprintf('\nit takes time about: %.3f s\n',timecost)
% % end
% % % matlabpool close
% % 
% % save ('feature_strong_weak1.mat','feature_strong','feature_weak')

% % matlabpool local 4
% parfor ii=4001:4500
%     fprintf('\nthe %dst image...........\n',ii);
% %     t = tic;
%  [feature_strong(ii,:), feature_weak(ii,:)]= extract_feature_forintegral(path,suffix,imnames,ii);
% %  fea  ture_strong(ii,:) = strong;
% %  feature_weak(ii,:) = weak;
%  
% %  imwrite([im_feature_dir,num2str(ii)])
% %     timecost = toc(t);
% %     fprintf('\nit takes time about: %.3f s\n',timecost)
% end
% % matlabpool close
% 
% save ('feature_strong_weak1.mat','feature_strong','feature_weak')
% 
% % matlabpool local 4
% parfor ii=4501:5000
%     fprintf('\nthe %dst image...........\n',ii);
% %     t = tic;
%  [feature_strong(ii,:), feature_weak(ii,:)]= extract_feature_forintegral(path,suffix,imnames,ii);
% %  fea  ture_strong(ii,:) = strong;
% %  feature_weak(ii,:) = weak;
%  
% %  imwrite([im_feature_dir,num2str(ii)])
% %     timecost = toc(t);
% %     fprintf('\nit takes time about: %.3f s\n',timecost)
% end
% % matlabpool close
% 
% save ('feature_strong_weak1.mat','feature_strong','feature_weak')
% end
%% select traindata and comparing with ground truth

%  ind = randperm( length(imnames) );
file_train = '.\jiang_data_select\arxiv_train.txt';
file_test = '.\jiang_data_select\arxiv_test.txt';
trainimages = mydataread(file_train);
testimages = mydataread(file_test);

% if 0
% % 获取训练图像、测试图像的在MSRA中的位置标签，便于从feature_XX中抽取特征数据
% train_location = [];
% test_location = [];
% t = tic;
% for ii=1:length(imnames)
%     tmp = imnames(ii).name(1:end-4);
%     
%     for ii_tr=1:length(trainimages)
%         tmp_tr = trainimages{ii_tr}(1:end-4);
%         if strcmp(tmp_tr,tmp)==1
%             train_location = [train_location;ii];
%         end
%         
%         clear tmp_tr
%     end
%     
%     for ii_te=1:length(testimages)
%         tmp_te = testimages{ii_te}(1:end-4);
%         if strcmp(tmp_te,tmp)==1
%             test_location = [test_location;ii];
%         end
%         
%         clear tmp_te
%     end
%     
%     clear tmp
% end
% time_cost = toc(t);
% end

if 0
% 根据AUC(Fmeasure)获取对应标签  1  0
extname_weak = suffix.weak;
[recall_avg_weak, precision_avg_weak,fpr_avg_weak, tpr_avg_weak ,auc_curve_weak,aucs_weak] = ...
    myCurveCompute(imgRoot_gt,imgRoot_weaksal,extname_weak);

extname_strong = suffix.strong;
[recall_avg_strong, precision_avg_strong ,fpr_avg_strong ,tpr_avg_strong ,auc_curve_strong,aucs_strong] = ...
    myCurveCompute(imgRoot_gt,imgRoot_strongsal,extname_strong);

save('weak_strong_PR_ROC.mat','recall_avg_weak','precision_avg_weak','fpr_avg_weak','tpr_avg_weak','auc_curve_weak','aucs_weak', ...
    'recall_avg_strong','precision_avg_strong' ,'fpr_avg_strong' ,'tpr_avg_strong' ,'auc_curve_strong','aucs_strong')

% 质量评价标签
qualitylabels_weak = single(aucs_weak >= aucs_strong);
qualitylabels_strong = 1 - qualitylabels_weak;

% 抽取数据(特征加标签) strong+weak
load ('.\jiang_data_select\arxiv_tr_te_location.mat')

traindata = [feature_strong(train_location,:);feature_weak(train_location,:)];
trainlabel = [qualitylabels_strong(train_location);qualitylabels_weak(train_location)];

testdata = [feature_strong(test_location,:);feature_weak(test_location,:)];
testlabel = [qualitylabels_strong(test_location);qualitylabels_weak(test_location)];
save('raw_feature.mat','traindata','trainlabel','testdata','testlabel')

end
load raw_feature.mat
[trainscale,validscale] = scaleForSVM_corrected(traindata,testdata,0,1);

clear traindata testdata
clear feature_strong feature_weak qualitylabels_weak qualitylabels_strong
%% trainning and obtain the evaluate model and testing obtain the decvalue as integrate coefficients
% model = train_integration_model( traindata,trainlabel );
% [predict_labels,accuracy,decvalue] = predict_integration_coffients(model,testdata,testlabel);

% s = '-t 0';% 线性核
% m = svmtrain(trainlabel, traindata, s);
% [pred_l, acc, dec_v] = svmpredict(testlabel, testdata, m);

[pred_l, acc , dec_v] = svm_onevsall(trainlabel,trainscale,testlabel,validscale);

% 识别距离归一化
dec_v(:,1) = (dec_v(:,1) - min(dec_v(:,1)))/max(max(dec_v(:,1)) - min(dec_v(:,1)),eps);
dec_v(:,2) = (dec_v(:,2) - min(dec_v(:,2)))/max(max(dec_v(:,2)) - min(dec_v(:,2)),eps);

% 找出weak strong各自的得分
[dec_v1,predict_labels] = max(dec_v, [], 2);

clear trainscale trainlabel testlabel 
%% integration
% 分配质量评价得分
load ('.\jiang_data_select\arxiv_tr_te_location.mat')
score_strong = dec_v1(1:length(test_location),:);
score_weak = dec_v1(length(test_location)+1:end,:);

% 计算出融合选择系数
Astrong = score_strong./max((score_strong + score_weak),eps);
Aweak = score_weak./max((score_strong + score_weak),eps);

suffix.weak = '_SelSalPixel.png';
suffix.strong = '_strong_filter.png';
% 开始融合:开并行，多核计算加快速度
% matlabpool local 4
parfor ii=1:length(testimages)
    disp(test_location(ii))
    tmpname = testimages{ii}(1:end-4);
    astrong = Astrong(ii);
    aweak = Aweak(ii);
    % 采用两种融合方法
    quality_biased_integrate(path,tmpname,astrong,aweak,suffix);
end

% matlabpool close

%% game over
msgbox('well done,boy,integration !!!')






