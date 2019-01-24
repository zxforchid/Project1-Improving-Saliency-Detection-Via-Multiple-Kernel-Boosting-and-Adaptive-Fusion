% This make.m is used under Windows

% mex -O -c svm.cpp
% mex -O -c svm_model_matlab.c
% mex -O svmtrain.c svm.obj svm_model_matlab.obj
% mex -O svmpredict.c svm.obj svm_model_matlab.obj
% mex -O libsvmread.c
% mex -O libsvmwrite.c

		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims svmtrain.c ./svm.cpp svm_model_matlab.c
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims svmpredict.c ./svm.cpp svm_model_matlab.c