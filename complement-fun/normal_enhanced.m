function  result = normal_enhanced(sp_sal_prob)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������һ�ι�һ��
% ������enhance
% �������һ�ι�һ��
% 07/15/2014
% xiaofei zhou,shanghai university
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��һ��
% sp_sal_prob = (sp_sal_prob - min(sp_sal_prob(:))) /...
%       (max(sp_sal_prob(:)) - min(sp_sal_prob(:)) + eps);
   
% enhanced
sp_sal_prob = exp( 1.25 * sp_sal_prob );
% sp_sal_prob = exp( sp_sal_prob );
% ��һ��
result = (sp_sal_prob - min(sp_sal_prob(:))) /...
      (max(sp_sal_prob(:)) - min(sp_sal_prob(:)) + eps);

clear sp_sal_prob

end