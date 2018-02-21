all:
	nvcc -O -I$CUDA_INSTALL_PATH/samples/common/inc matrixKernel.cu -L$CUDA_INSTALL_PATH/samples/common/lib/linux/x86_64 -lcurand -lcublas

clean:
	rm ./a.out
