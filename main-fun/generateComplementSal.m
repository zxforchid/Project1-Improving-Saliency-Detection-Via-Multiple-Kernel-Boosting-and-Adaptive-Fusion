function [salComplement, salComplementFilter] = ...
    generateComplementSal(Thigh, Tlow, colorImname,initialImgname,saveComplementPath,superPath)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1: 2016/1/27
% COPYRIGHT(C) by xiaofei zhou,VIPL, shanghai university,shanghai,china
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im=imread(colorImname);
salInitial = double(imread(initialImgname));
[w,h,dim]=size(im);
if dim==1
   temp=zeros(w,h,3);
   temp(:,:,1)=im;
   temp(:,:,2)=im;
   temp(:,:,3)=im;
   im=temp;
end

%% cut frame &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
[input_im,flag_strong]=cutframe(im);
input_im=uint8(input_im);
salInitialNorm = normalize(double(salInitial));

%% variable initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
[m,n,k] = size(input_im);
colorImnameSlic=[superPath,colorImname(1:end-4) '.bmp'];
imwrite(input_im,colorImnameSlic);

% feature data initialize
datap.rgb=[];datan.rgb=[];
datap.lab=[];datan.lab=[];
datap.hsv=[];datan.hsv=[];
datap.lm=[];datan.lm=[];
datap.lbp=[];datan.lbp=[];
datap.sift=[];datan.sift=[];
datap.hog=[];datan.hog=[];
datap.geodist=[];datan.geodist=[];

% label:positive and negative
labelp=[];labeln=[];

% all samples features
rgbvals0=[];labvals0=[];hsvvals0=[];
lmvals0=[];lbpvals0=[];siftvals0=[];
hogvals0=[];geodistvals0=[];

%% filter initial for graph cut &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
disp('building graph');
N = m*n;
imm = double((input_im));
m1=imm(:,:,1);
m2=imm(:,:,2);
m3=imm(:,:,3);

% construct graph
E = edges4connected(m,n);
V=1./(1+sqrt((m1(E(:,1))-m1(E(:,2))).^2+(m2(E(:,1))-m2(E(:,2))).^2+(m3(E(:,1))-m3(E(:,2))).^2));
AA=1000*sparse(E(:,1),E(:,2),0.3*V);
g = fspecial('gauss', [5 5], sqrt(5));

%% multiscale feature extract %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare data for feature generation
imdata = drfiGetImageData( input_im );
for ss=2:5
    spnumber=(ss-0)*50;
%% generate superpixels
   % the slic software support only the '.bmp' image
    comm=['SLICSuperpixelSegmentation' ' ' colorImnameSlic ' ' int2str(20) ' ' int2str(spnumber) ' ' superPath];
    system(comm);    
    spname=[superPath colorImname(1:end-4)  '.dat'];
    superpixels=ReadDAT([m,n],spname);
    spnum=max(superpixels(:));
    imsegs = processSuperpixelImage_forSLIC(superpixels,spnum);
    boundsp_bg = extract_bg_sp(superpixels,m,n); 
%% compute the feature
% data of the pesudo-background
pbgdata.label = boundsp_bg;

% data of each superpixel
[spdata,STA] = drfiGetSuperpixelData2( imdata, imsegs );

if ss==2
    STA0=STA;
else if ss==3
        STA1=STA;
    else if ss==4
            STA2=STA;
        else
            STA3=STA;
        end
    end
end

% saliency feature of each segment (region)
featuredata = drfiGetRegionSaliencyFeature2( imsegs, spdata, imdata, pbgdata );
 
%% use reference as weak saliency model
    bst=unique(superpixels(1,1:n));
    bsd=unique(superpixels(m,1:n));
    bsr=unique(superpixels(1:m,1));
    bsl=unique(superpixels(1:m,n));
    bs=sort(unique([bst,bsd,bsr',bsl']));
    
 %% MKB
    bsf=[];
    bsb=[];
    % high threshold and low threshold are needed to compute better
    ThighValue=Thigh .* mean(salInitialNorm(:));
    TlowValue=Tlow .* mean(salInitialNorm(:));
    maxi=0;
    for i=1:spnum 
        meantemp =mean(salInitialNorm(STA(i).PixelIdxList));
        if meantemp>maxi
            maxi=meantemp;
            maxind=i;
        end
        if meantemp>= ThighValue
            bsf=[bsf,i];% 正样本
        else if meantemp< TlowValue
            bsb=[bsb,i];% 负样本
            end
        end 
        
    end
    if isempty(bsf)
       bsf=[maxind]; 
    end
       bsb=unique([bsb,bs]);
       
 datap.rgb=[datap.rgb;featuredata.rgb(bsf,:)];
 datan.rgb=[datan.rgb;featuredata.rgb(bsb,:)];
 datap.lab=[datap.lab;featuredata.lab(bsf,:)];
 datan.lab=[datan.lab;featuredata.lab(bsb,:)];
 datap.hsv=[datap.hsv;featuredata.hsv(bsf,:)];
 datan.hsv=[datan.hsv;featuredata.hsv(bsb,:)];
 datap.lm=[datap.lm;featuredata.lm(bsf,:)];
 datan.lm=[datan.lm;featuredata.lm(bsb,:)];
 datap.lbp=[datap.lbp;featuredata.lbp(bsf,:)];
 datan.lbp=[datan.lbp;featuredata.lbp(bsb,:)];
 datap.sift=[datap.sift;featuredata.sift(bsf,:)];
 datan.sift=[datan.sift;featuredata.sift(bsb,:)];
 datap.hog=[datap.hog;featuredata.hog(bsf,:)];
 datan.hog=[datan.hog;featuredata.hog(bsb,:)];
 datap.geodist=[datap.geodist;featuredata.geodist(bsf,:)];
 datan.geodist=[datan.geodist;featuredata.geodist(bsb,:)];
 
labelp=[labelp;repmat(1,length(bsf),1)];
labeln=[labeln;repmat(-1,length(bsb),1)];

rgbvals0=[rgbvals0;featuredata.rgb];
labvals0=[labvals0;featuredata.lab];
hsvvals0=[hsvvals0;featuredata.hsv];
lmvals0=[lmvals0;featuredata.lm];
lbpvals0=[lbpvals0;featuredata.lbp];
siftvals0=[siftvals0;featuredata.sift];
hogvals0=[hogvals0;featuredata.hog];
geodistvals0=[geodistvals0;featuredata.geodist];

clear featuredata

end
%% beagin training and testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
traindata.rgb=[datap.rgb;datan.rgb];
traindata.lab=[datap.lab;datan.lab];
traindata.hsv=[datap.hsv;datan.hsv];
traindata.lm=[datap.lm;datan.lm];
traindata.lbp=[datap.lbp;datan.lbp];
traindata.sift=[datap.sift;datan.sift];
traindata.hog=[datap.hog;datan.hog];
traindata.geodist=[datap.geodist;datan.geodist];

 testdata.rgb=rgbvals0;
 testdata.lab=labvals0;
 testdata.hsv=hsvvals0;
 testdata.lm=lmvals0;
 testdata.lbp=lbpvals0;
 testdata.sift=siftvals0;
 testdata.hog=hogvals0;
 testdata.geodist=geodistvals0;
 
label.rgb = [labelp;labeln];label.lab = label.rgb;
label.lbp = label.rgb;label.hsv = label.rgb;
label.lm = label.rgb;label.sift = label.rgb;
label.hog = label.rgb;label.geodist = label.rgb;

 clear datap datan rgbvals0 labvals0 hsvvals0 lmvals0 lbpvals0 siftvals0 geodistvals0 hogvals0
 
 
 % 归一化 scaleForSVM_corrected scaleForSVM
 [traindata.rgb,testdata.rgb]         = scaleForSVM_corrected(traindata.rgb,testdata.rgb,0,1);
 [traindata.lab,testdata.lab]         = scaleForSVM_corrected(traindata.lab,testdata.lab,0,1);
 [traindata.hsv,testdata.hsv]         = scaleForSVM_corrected(traindata.hsv,testdata.hsv,0,1);
 [traindata.lm,testdata.lm]           = scaleForSVM_corrected(traindata.lm,testdata.lm,0,1);
 [traindata.lbp,testdata.lbp]         = scaleForSVM_corrected(traindata.lbp,testdata.lbp,0,1);
 [traindata.sift,testdata.sift]       = scaleForSVM_corrected(traindata.sift,testdata.sift,0,1);
 [traindata.hog,testdata.hog]         = scaleForSVM_corrected(traindata.hog,testdata.hog,0,1);
 [traindata.geodist,testdata.geodist] = scaleForSVM_corrected(traindata.geodist,testdata.geodist,0,1);
 
% 获取特征对应样本数量
datanum = [size(traindata.rgb,1) size(traindata.lab,1) size(traindata.hsv,1) ...
    size(traindata.lm,1) size(traindata.lbp,1) size(traindata.sift,1) ...
    size(traindata.hog,1) size(traindata.geodist,1)];
%% %%%%%%%%%%%%%%%%%%%% Trainning and Testing %%%%%%%%%%%%%%%%%%%%
errormethod = 'MSE';
nfeature = 8;ntype = 4;
fprintf('\ntrainning....................................................\n')
[ beta, model, tm ] = boost_SVR3( traindata, label, nfeature, datanum,ntype,errormethod);
 d = distribution( model );  

 tic
 fprintf('\ntesting....................................................\n')
[conf] = mkltest_forRegression1(beta, model,d,testdata,ntype);
clear traindata testdata
toc

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 map0=zeros(m,n);  
 map1=zeros(m,n);
 map2=zeros(m,n);
 map3=zeros(m,n);

 spnum0=size(STA0,1);
 spnum1=size(STA1,1);
 spnum2=size(STA2,1); 
 spnum3=size(STA3,1);
 
  for i=1:spnum0
      map0(STA0(i).PixelIdxList)=repmat(conf(i), size(STA0(i).PixelIdxList));  
  end
   for i=spnum0+1:spnum0+spnum1
      map1(STA1(i-spnum0).PixelIdxList)=repmat(conf(i), size(STA1(i-spnum0).PixelIdxList));  
   end
   for i=spnum0+spnum1+1:spnum0+spnum1+spnum2
      map2(STA2(i-spnum0-spnum1).PixelIdxList)=repmat(conf(i), size(STA2(i-spnum0-spnum1).PixelIdxList));  
   end
   for i=spnum0+spnum1+spnum2+1:spnum0+spnum1+spnum2+spnum3
      map3(STA3(i-spnum0-spnum1-spnum2).PixelIdxList)=repmat(conf(i), size(STA3(i-spnum0-spnum1-spnum2).PixelIdxList));  
   end
  
   clear conf
   
   map0=normalize(map0);
   map1=normalize(map1);
   map2=normalize(map2);
   map3=normalize(map3);
  
  [mkbcut0]=graphcut0(AA,g,map0);
  [mkbcut1]=graphcut0(AA,g,map1);
  [mkbcut2]=graphcut0(AA,g,map2);
  [mkbcut3]=graphcut0(AA,g,map3);
  mkbcut=normalize(mkbcut0+mkbcut1+mkbcut2+mkbcut3);
  
salComplement=saveimg(flag_strong,mkbcut,[w,h]);
salComplementFilter= guidedfilter(salComplement,salComplement,7,0.1);
salComplementFilter = normalize(salComplementFilter);
clear mkbcut0 mkbcut1 mkbcut2 mkbcut3 map0 map1 map2 map3

salComplementForwrite = uint8(255 * salComplement);
imwrite(salComplementForwrite,[saveComplementPath, colorImname(1:end-4), '_complement.png']);

salComplementFilterForwrite = uint8( 255 * salComplementFilter);
imwrite(salComplementFilterForwrite,[saveComplementPath, colorImname(1:end-4), '_complement_filter.png']); 

clear salComplementFilterForwrite salComplementForwrite
end



