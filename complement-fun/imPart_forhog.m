% 2015.7.9 19:42
% by Xiaofei Zhou
% ���ݹ̶������ֽ�ͼ��
% patchSize=[p,q]
function [imgout] = imPart_forhog(imgin,patchSize,cellsize)
imgin = double(imgin);


if sum(patchSize)>2

p = patchSize(1);
q = patchSize(2);
imgout = cell(p,q);% �洢ʱ���н��д洢�������Ͷ�Ӧ������
% ��ͼ��imginƽ���ķֳ�numOfpatches�飺���з�ʽ����
[m, n] = size(imgin);
% patch_sizem=floor(m/p);
% patch_sizen=floor(n/q);
patch_sizem=cellsize;
patch_sizen=cellsize;
count=0;
numx=0;
for x = 1:patch_sizem:(p-1)*patch_sizem+1
    numx=numx+1;numy=0;
    for y = 1:patch_sizen:(q-1)*patch_sizen+1
        count=count+1;
        numy=numy+1;
        if numy==q && numx<p % �����Ե,�ϲ������Ԫ��
            imgout{numx,numy}=imgin(x:x+patch_sizem-1,y:end);
        elseif numx==p && numy<q % �����Ե,�ϲ������Ԫ��
            imgout{numx,numy}=imgin(x:end,y:y+patch_sizen-1);
        elseif numx==p && numy==q % �����Ե,�ϲ������Ԫ��
            imgout{numx,numy}=imgin(x:end,y:end);
        else
            imgout{numx,numy}=imgin(x:x+patch_sizem-1,y:y+patch_sizen-1);% �����ֽ�,�ϲ������Ԫ��
        end 
    end
end 

end 
return
    