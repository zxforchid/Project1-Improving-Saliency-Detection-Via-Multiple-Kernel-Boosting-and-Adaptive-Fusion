% compile
% 07/08/2015
% written by xiaofei zhou,shanghai university,shanghai,china
% 

  mex private/edgesDetectMex.cpp -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/edgesNmsMex.cpp    -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/spDetectMex.cpp    -outdir private '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  mex private/edgeBoxesMex.cpp   -outdir private