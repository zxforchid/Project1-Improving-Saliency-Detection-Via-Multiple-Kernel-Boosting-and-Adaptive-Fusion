function result = drfiGetRegionSaliencyFeature2( imsegs, spdata, imdata, pbgdata )  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1:
% 修改原始LU HUCHUAN程序，使之变成对比度
% [global,local,backgroundnes]
% 并且同类性质的特征连接在一起
% copyright by xiaofei zhou,shanghai university,shanghai,china
% current version: 07/07/2015 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %% initial 
    nseg = imsegs.nseg;        
    spstats = regionprops( imsegs.segimage, 'Centroid', 'PixelIdxList', 'Area', 'Perimeter' );  
    adjmat = double( imsegs.adjmat ) .* (1 - eye(nseg, nseg));
    
    r = double( imdata.image_rgb(:,:,1) );
    g = double( imdata.image_rgb(:,:,2) );
    b = double( imdata.image_rgb(:,:,3) );
    L = imdata.image_lab(:,:,1);
    a = imdata.image_lab(:,:,2);
    bb = imdata.image_lab(:,:,3);
    h = imdata.image_hsv(:,:,1);
    s = imdata.image_hsv(:,:,2);
    v = imdata.image_hsv(:,:,3);
    
    %% image size and superpixel size and background size
    [imh imw] = size(r);
    [~,nsp] = size(spdata.R);
    [~,npb] = size(pbgdata.label);
    
    %% position and area information
    position = zeros(nseg, 2);
    area = zeros(1, nseg);
    
    for ix = 1 : length(spstats)                   
        position(ix, :) = (spstats(ix).Centroid);
        area(ix) = spstats(ix).Area;
    end
    position = position / max(imh, imw);% 距离归一化
    
    %% global and local and boundary weight
    area_adj_weight = repmat(area, [nseg, 1]) .* adjmat;
    area_adj_weight = area_adj_weight ./ repmat(sum(area_adj_weight, 2) + eps, [1, nseg]);  
    
    area_all_weight = repmat(area, [nseg, 1])./repmat(sum(area,2)+eps,[nseg,nseg]);
    
    boundary = zeros(nseg,nseg);
    boundary(:,pbgdata.label) = 1;
    area_boundary_weight = boundary.*area_all_weight;
    
    sp = 0.5; % sp = 1 / 0.4;%0.5 / ( 0.25 * 0.25 );
    dp = mexFeatureDistance( position', [], 'L2' );
    
    dist_weight_global = area_all_weight.*exp( -sp * dp );% global weight
    dist_weight_local = area_adj_weight.*exp( -sp * dp .* adjmat);% local weight:相邻接则不为零，否则，为零
    dist_weight_boundary = area_boundary_weight.*exp(-sp*dp.*boundary);% boundary weight: 边界位置不为零，非边界位置为零
    
    %% feature initialize: global local boundary
    result = struct;
    result.rgb = zeros(nseg,21);%(3+1+1)*3 + 3 + 3= 21 
    result.lab = zeros(nseg,21);
    result.hsv = zeros(nseg,21);
    result.lm = zeros(nseg,93);% (15+1)*3 + 15 = 93
    result.lbp = zeros(nseg,63);% 59+1*3 + 1 = 63    
    result.sift = zeros(nseg,259);% 1*3+128+128=259
    result.hog = zeros(nseg,67);% 1*3 + 32 + 32 = 67
    result.geodist = zeros(nseg,5);% 1*3 + 1 + 1 = 5
%     result.objectness;
    %% begin to computer features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% region contrast (all)
    feat_dist_mat = zeros(nseg, nseg, 35);
    % mean R, G, B distance, and x2 distance of RGB histogram
    feat_dist_mat(:,:,1) = mexFeatureDistance(spdata.R, [], 'L1');
    feat_dist_mat(:,:,2) = mexFeatureDistance(spdata.G, [], 'L1');
    feat_dist_mat(:,:,3) = mexFeatureDistance(spdata.B, [], 'L1');
    feat_dist_mat(:,:,4) = sqrt(feat_dist_mat(:,:,1).^2 + feat_dist_mat(:,:,2).^2 ...
        + feat_dist_mat(:,:,3).^2);
    feat_dist_mat(:,:,5) = mexFeatureDistance(spdata.RGBHist, [], 'x2');    
     
    % mean L, a, b distance, and x2 distance of Lab histogram
    feat_dist_mat(:,:,6) = mexFeatureDistance(spdata.L, [], 'L1');
    feat_dist_mat(:,:,7) = mexFeatureDistance(spdata.a, [], 'L1');
    feat_dist_mat(:,:,8) = mexFeatureDistance(spdata.b, [], 'L1');    
    feat_dist_mat(:,:,9) = sqrt(feat_dist_mat(:,:,6).^2 + feat_dist_mat(:,:,7).^2 ...
        + feat_dist_mat(:,:,8).^2);
    feat_dist_mat(:,:,10) = mexFeatureDistance(spdata.LabHist, [], 'x2');
    
    % mean H, S, V distance, and x2 distance of HSV histogram
    feat_dist_mat(:,:,11) = mexFeatureDistance(spdata.H, [], 'L1');
    feat_dist_mat(:,:,12) = mexFeatureDistance(spdata.S, [], 'L1');
    feat_dist_mat(:,:,13) = mexFeatureDistance(spdata.V, [], 'L1'); 
    feat_dist_mat(:,:,14) = sqrt(feat_dist_mat(:,:,11).^2 + feat_dist_mat(:,:,12).^2 ...
        + feat_dist_mat(:,:,13).^2);
    feat_dist_mat(:,:,15) = mexFeatureDistance(spdata.HSVHist, [], 'x2');
    
    % LM 15
    for ix = 1 : imdata.ntext
        feat_dist_mat(:,:,15+ix) = mexFeatureDistance(spdata.texture(ix,:), [], 'L1');
    end
    
    feat_dist_mat(:,:,31) = mexFeatureDistance(spdata.textureHist, [], 'x2');
    
    % LBP
    feat_dist_mat(:,:,32) = mexFeatureDistance(spdata.lbpHist, [], 'x2');
  
    % SIFT 欧氏距离度量
    feat_dist_mat(:,:,33) = GetDistanceMatrix((spdata.sift)');
    
    % HOG 欧氏距离度量
    feat_dist_mat(:,:,34) = GetDistanceMatrix((spdata.hog)');
    
    % Geodestic
    feat_dist_mat(:,:,35) = mexFeatureDistance(spdata.geoDist, [], 'L1');
    
    %% regional variance:RGB/LAB/HSV LM LBP SIFT HOG
    feat_dist_mat_property = zeros(nsp,25+128+32+1);
    for reg = 1 : nsp
        pixels = spstats(reg).PixelIdxList;
        
        feat_dist_mat_property(reg,1:3) = [var( r(pixels) ), var( g(pixels) ), var( b(pixels) )];% RGB var
        feat_dist_mat_property(reg,4:6) = [var( L(pixels) ), var( a(pixels) ), var( bb(pixels) )];% LAB var
        feat_dist_mat_property(reg,7:9) = [var( h(pixels) ), var( s(pixels) ), var( v(pixels) )];% HSV var
        
        % LM var
        for it = 1 : imdata.ntext
            temp_text = imdata.imtext(:,:,it);
            feat_dist_mat_property(reg, 9+it) = var( temp_text(pixels) );
        end
        
        % LBP var
        feat_dist_mat_property(reg,25) = var( imdata.imlbp(pixels) );
       
        % SIFT var
        for isift = 1 : imdata.nsift
            temp_text = imdata.sift(:,:,isift);
            feat_dist_mat_property(reg, 25+isift) = var( temp_text(pixels) );
        end
        
        % HOG var
        for ihog = 1 : imdata.nhog
            temp_text = imdata.hog(:,:,ihog);
            feat_dist_mat_property(reg, 25+imdata.nsift+ihog) = var( temp_text(pixels) );
        end
        
        % Geodestic Var
        feat_dist_mat_property(reg,186) = var( imdata.geoDist(pixels) );
        
    end
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% regional contrast:global and local and boundary
    %% rgb/hsv/lab ----------------------------------------------
    for icl=1:5
        % global
        result.rgb(:,icl) = sum(feat_dist_mat(:,:,icl) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);     
        result.lab(:,icl) = sum(feat_dist_mat(:,:,icl+5) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps); 
        result.hsv(:,icl) = sum(feat_dist_mat(:,:,icl+10) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps); 
        
        % local
        result.rgb(:,icl+5) = sum(feat_dist_mat(:,:,icl) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);     
        result.lab(:,icl+5) = sum(feat_dist_mat(:,:,icl+5) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps); 
        result.hsv(:,icl+5) = sum(feat_dist_mat(:,:,icl+10) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps); 
        
        % boundary
        result.rgb(:,icl+10) = sum(feat_dist_mat(:,:,icl) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);     
        result.lab(:,icl+10) = sum(feat_dist_mat(:,:,icl+5) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps); 
        result.hsv(:,icl+10) = sum(feat_dist_mat(:,:,icl+10) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps); 
    end
    result.rgb(:,16:end) = [feat_dist_mat_property(:,1:3),(spdata.R)',(spdata.G)',(spdata.B)'];
    result.lab(:,16:end) = [feat_dist_mat_property(:,4:6),(spdata.L)',(spdata.a)',(spdata.b)'];
    result.hsv(:,16:end) = [feat_dist_mat_property(:,7:9),(spdata.H)',(spdata.S)',(spdata.V)'];
    
    %% LM 93
    for ilm=1:imdata.ntext
        result.lm(:,ilm)    = sum(feat_dist_mat(:,:,15+ilm) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);        
        result.lm(:,ilm+15) = sum(feat_dist_mat(:,:,15+ilm) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);        
        result.lm(:,ilm+30) = sum(feat_dist_mat(:,:,15+ilm) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);
    end
    result.lm(:,46) = sum(feat_dist_mat(:,:,31) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);
    result.lm(:,47) = sum(feat_dist_mat(:,:,31) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);
    result.lm(:,48) = sum(feat_dist_mat(:,:,31) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);    
    result.lm(:,49:end) = [feat_dist_mat_property(:,10:24),(spdata.texture)',(spdata.textureHist)'];
    
    %% LBP 63
    result.lbp(:,1) = sum(feat_dist_mat(:,:,32) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);
    result.lbp(:,2) = sum(feat_dist_mat(:,:,32) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);
    result.lbp(:,3) = sum(feat_dist_mat(:,:,32) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);
    result.lbp(:,4) = feat_dist_mat_property(:,25);
    result.lbp(:,5:end) = (spdata.lbpHist)';
    
    %% SIFT 259
    result.sift(:,1) = sum(feat_dist_mat(:,:,33) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);
    result.sift(:,2) = sum(feat_dist_mat(:,:,33) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);
    result.sift(:,3) = sum(feat_dist_mat(:,:,33) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);
    result.sift(:,4:131) = feat_dist_mat_property(:,26:153);
    result.sift(:,132:end) = (spdata.sift)';
    
    %% HOG 67
    result.hog(:,1) = sum(feat_dist_mat(:,:,34) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);
    result.hog(:,2) = sum(feat_dist_mat(:,:,34) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);
    result.hog(:,3) = sum(feat_dist_mat(:,:,34) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);
    result.hog(:,4:35) = feat_dist_mat_property(:,154:185);% 32
    result.hog(:,36:end) = (spdata.hog)';% 32
    
    %% geodestic 5   
    result.geodist(:,1) = sum(feat_dist_mat(:,:,35) .* dist_weight_global, 2) ./ (sum(dist_weight_global, 2) + eps);
    result.geodist(:,2) = sum(feat_dist_mat(:,:,35) .* dist_weight_local, 2) ./ (sum(dist_weight_local, 2) + eps);
    result.geodist(:,3) = sum(feat_dist_mat(:,:,35) .* dist_weight_boundary, 2) ./ (sum(dist_weight_boundary, 2) + eps);
    result.geodist(:,4:end) = [feat_dist_mat_property(:,186),(spdata.geoDist)'];
    %% clear variables
    clear feat_dist_mat feat_dist_mat_property 
    clear imsegs spdata imdata pbgdata
end
