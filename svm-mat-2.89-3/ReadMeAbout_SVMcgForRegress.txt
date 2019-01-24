[mse,bestc,bestg] = SVMcgForRegress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)

train_label:训练集标签.要求与libsvm工具箱中要求一致.
train:训练集.要求与libsvm工具箱中要求一致.
cmin:惩罚参数c的变化范围的最小值(取以2为底的对数后),即 c_min = 2^(cmin).默认为 -5
cmax:惩罚参数c的变化范围的最大值(取以2为底的对数后),即 c_max = 2^(cmax).默认为 5
gmin:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmin).默认为 -5
gmax:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmax).默认为 5

v:cross validation的参数,即给测试集分为几部分进行cross validation.默认为 3
cstep:参数c步进的大小.默认为 1
gstep:参数g步进的大小.默认为 1
msestep:最后显示准确率图时的步进大小. 默认为 20
