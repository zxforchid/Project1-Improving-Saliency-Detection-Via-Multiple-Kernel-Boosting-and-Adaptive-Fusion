function  result = normal_enhanced(sp_sal_prob)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 首先做一次归一化
% 接着做enhance
% 最后再做一次归一化
% 07/15/2014
% xiaofei zhou,shanghai university
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 归一化
% sp_sal_prob = (sp_sal_prob - min(sp_sal_prob(:))) /...
%       (max(sp_sal_prob(:)) - min(sp_sal_prob(:)) + eps);
   
% enhanced
sp_sal_prob = exp( 1.25 * sp_sal_prob );
% sp_sal_prob = exp( sp_sal_prob );
% 归一化
result = (sp_sal_prob - min(sp_sal_prob(:))) /...
      (max(sp_sal_prob(:)) - min(sp_sal_prob(:)) + eps);

clear sp_sal_prob

end