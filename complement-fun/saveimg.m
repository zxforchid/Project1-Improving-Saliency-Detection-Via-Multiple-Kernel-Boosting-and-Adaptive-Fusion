function [saliency]= saveimg(flag,sal,sizeim)
w=sizeim(1);
h=sizeim(2);
if(flag) 
saliency=zeros(w,h); 
saliency(flag+1:w-flag,flag+1:h-flag)=sal;
else
saliency=sal;
end
saliency=normalize(saliency);
end

