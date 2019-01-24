function normal = normalization(x, kind)
% by Fengyong Li

if nargin < 2
    kind = 2;
end
%----------------------------
[m, n] = size(x);
normal = zeros(m, n);
%---------------------------
%normalize the data x to [0, 1]
if kind == 1
    for i = 1:m
        ma = max( x(i,:) );
        mi = min( x(i,:) );
        normal(i,:) = ( x(i,:)-mi )./(ma-mi);
    end
end
%-----------------------------
%normalize the data x to [-1, 1]
if kind == 2
    for i = 1:m
        mea = mean( x(i,:) );
        va = var( x(i,:) );
         normal(i,:) = ( x(i,:)-mea )/va;
    end
end










