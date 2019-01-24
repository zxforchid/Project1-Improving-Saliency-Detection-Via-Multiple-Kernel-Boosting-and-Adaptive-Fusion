function [spdata,spstats] = drfiGetSuperpixelData2( imdata, imsegs )
% function [spdata,spstats] = drfiGetSuperpixelData2( imdata, imsegs ,pbgdata)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version1
% spstats 返回的区域超像素信息
% copyright by xiaofei zhou,shanghai university,shanghai,china
% 2015.7.9 20:51PM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     spstats = regionprops( imsegs.segimage, 'PixelIdxList' );
    spstats = regionprops( imsegs.segimage, 'all' );
    
    image_rgb = imdata.image_rgb;
    image_lab = imdata.image_lab;
    image_hsv = imdata.image_hsv;
    
    Q_rgb = imdata.Q_rgb;
    Q_lab = imdata.Q_lab;
    Q_hsv = imdata.Q_hsv;
    
    imtext = imdata.imtext;
    texthist = imdata.texthist;
    imlbp = imdata.imlbp;    
    imsift = imdata.sift;
    imhog = imdata.hog;
    imgeodist = imdata.geoDist;
    
    imw = imdata.imw;
    imh = imdata.imh;
    
    nseg = imsegs.nseg;
    
    spdata.R = zeros(1, nseg);
    spdata.G = zeros(1, nseg);
    spdata.B = zeros(1, nseg);
    
    spdata.RGBHist = zeros(imdata.nRGBHist, nseg);
    
    spdata.L = zeros(1, nseg);
    spdata.a = zeros(1, nseg);
    spdata.b = zeros(1, nseg);
    
    spdata.LabHist = zeros(imdata.nLabHist, nseg);
    
    spdata.H = zeros(1, nseg);
    spdata.S = zeros(1, nseg);
    spdata.V = zeros(1, nseg);
    
    spdata.HSVHist = zeros(imdata.nHSVHist, nseg);
    
    spdata.texture = zeros(imdata.ntext, nseg);
    
    spdata.textureHist = zeros(imdata.ntext, nseg);
    
    spdata.lbpHist = zeros(imdata.nlbp, nseg);
    
    spdata.sift = zeros(imdata.nsift, nseg);
    
    spdata.hog = zeros(imdata.nhog, nseg);
    
    spdata.geoDist = zeros(1,nseg); 
    %% compute region feature
    pixelList = cell(nseg, 1);
    for s = 1 : nseg
        pixels = spstats(s).PixelIdxList;
        pixelList{s} = pixels;
        spdata.R(s) = mean( image_rgb(pixels) );
        spdata.G(s) = mean( image_rgb(pixels + imw * imh) );
        spdata.B(s) = mean( image_rgb(pixels + imw * imh * 2) );
        
        spdata.RGBHist(:, s) = hist( Q_rgb(pixels), 1:imdata.nRGBHist )';
        spdata.RGBHist(:, s) = spdata.RGBHist(:, s) / max( sum(spdata.RGBHist(:, s)), eps );
        
        spdata.L(s) = mean( image_lab(pixels) );
        spdata.a(s) = mean( image_lab(pixels + imw * imh) );
        spdata.b(s) = mean( image_lab(pixels + imw * imh * 2) );
        
        spdata.LabHist(:, s) = hist( Q_lab(pixels), 1:imdata.nLabHist )';
        spdata.LabHist(:, s) = spdata.LabHist(:, s) / max( sum(spdata.LabHist(:, s)), eps );
        
        spdata.H(s) = mean( image_hsv(pixels) );
        spdata.S(s) = mean( image_hsv(pixels + imw * imh) );
        spdata.V(s) = mean( image_hsv(pixels + imw * imh * 2) );
        
        spdata.HSVHist(:, s) = hist( Q_hsv(pixels), 1:imdata.nHSVHist )';
        spdata.HSVHist(:, s) = spdata.HSVHist(:, s) / max( sum(spdata.HSVHist(:, s)), eps );
        
        %-----lm----%
        % LM每一层对应超像素区域的均值，NSEG*15
        for ift = 1 : imdata.ntext
            spdata.texture(ift, s) = mean( imtext(pixels+(ift-1)*imw*imh) );
        end
        
        % LM最大值图像的对应超像素区域的直方图 NSEG*15
        spdata.textureHist(:, s) = hist( texthist(pixels), 1:imdata.ntext )';
        spdata.textureHist(:, s) = spdata.textureHist(:, s) / max( sum(spdata.textureHist(:, s)), eps );
        
        % -- LBP ---%
        spdata.lbpHist(:, s) = hist( imlbp(pixels), 0:58 )';
        spdata.lbpHist(:, s) = spdata.lbpHist(:, s) / max( sum(spdata.lbpHist(:, s)), eps );
        
        % -- sift --%
        for isift=1:imdata.nsift
            temp = imsift(:,:,isift);
            spdata.sift(isift,s) = mean(temp(pixels));
        end
        
        %---- hog ---%
        for ihog=1:imdata.nhog
            temp = imhog(:,:,ihog);
            spdata.hog(ihog,s) = mean(temp(pixels));
        end
        
        %---- Geodestic ----%
        % 区域均值作为超像素区域测地距离
        spdata.geoDist(s) = mean( imgeodist(pixels) );

    end     
    
    % clear
    clear imdata imsegs pbgdata
end