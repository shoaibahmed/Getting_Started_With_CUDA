module add nvidia/7.5
nvcc -O -I$CUDA_INSTALL_PATH/samples/common/inc Framework_GL.cpp kernel_frame.cu -L$CUDA_INSTALL_PATH/samples/common/lib/linux/x86_64 -lGLEW -lglut -lGLU -lGL
rhrk-launch --title GPU --command bash --module nvidia/7.5 --mode vgl
