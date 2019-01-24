% compile
% 07/08/2015
% written by xiaofei zhou,shanghai university,shanghai,china
% 

  mex private/gradientMex.cpp    -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/imPadMex.cpp       -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/imResampleMex.cpp  -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/rgbConvertMex.cpp  -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/convConst.cpp      -outdir private 