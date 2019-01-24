function [ out,j ] = cutframe( im )
[w,h,dim]=size(im);
gray=im2double(rgb2gray(im));
j=0;
for jj=1:16
%j=1;
%while(1)
    box=gray(jj:w+1-jj,jj:h+1-jj);
    rim=[gray(jj,jj:h+1-jj),gray(w+1-jj,jj:h+1-jj),gray(jj:w+1-jj,jj)',gray(jj:w+1-jj,h+1-jj)'];
    if (max(rim)-min(rim))<0.3
       j=jj;
    end
    %j=j+1;
end
if j>=1
    I=zeros(w-2*j,h-2*j,dim);
    for k=1:dim
   image=im(:,:,k);
    I(:,:,k)=image(j+1:w-j,j+1:h-j);
    end
out=I;
else
    out=im;
end;
end

