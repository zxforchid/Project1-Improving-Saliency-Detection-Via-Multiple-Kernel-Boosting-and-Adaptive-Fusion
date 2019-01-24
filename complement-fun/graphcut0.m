function [ fiimg ] = graphcut0( A,g,prior)
N=size(prior,1)*size(prior,2);
prior= max(prior, abs(imfilter(prior, g, 'symmetric')));
prior=normalize(prior); 
T=sparse(N,2);
T(:,1)=reshape(prior,[N,1]);
T(:,2)=1-reshape(prior,[N,1]);
disp('calculating maximum flow');
[height,width]=size(prior);
[flow,labels] = maxflow(A,T);
labels = reshape(labels,[height width]);
map=1-reshape(labels,height,width);
  fiimg=normalize(prior+double(map)); 
  side=[map(1:end,1)',map(1:end,end)',map(1,1:end),map(end,1:end)];
   if mean(side)>0.2 || mean(prior(:))>0.8
       fiimg=prior;
   end
   
end


