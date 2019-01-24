function prob = distribution_prob( d, sample, j )
%%
n = size(sample,1);
prob = zeros(n,1);
for i = 1:n
%     p1 = exp(-0.5 * (sample(i,:) - d{j,1}) * inv(d{j,2}) * (sample(i,:) - d{j,1})');
%     p2 = exp(-0.5 * (sample(i,:) - d{j,3}) * inv(d{j,4}) * (sample(i,:) - d{j,3})');
    p1 = exp(-sqrt(sum((sample(i,:)-d{j,1}).^2)));
    p2 = exp(-sqrt(sum((sample(i,:)-d{j,3}).^2)));
%     l = log (p1 / (p2+0.00001));
    l = log(p1/(p2+eps));
    prob(i) = 1 - exp(-abs(l));
end