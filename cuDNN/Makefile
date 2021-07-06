all:
	/usr/local/cuda/bin/nvcc -o cnn cnn.cpp -O2 -I/usr/local/cuda/include/ -I/usr/include/opencv4/ -L/usr/local/cuda/lib64/ -lcudnn -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	/usr/local/cuda/bin/nvcc -o cnn_nd cnn_nd.cpp -O2 -I/usr/local/cuda/include/ -I/usr/include/opencv4/ -L/usr/local/cuda/lib64/ -lcudnn -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	/usr/local/cuda/bin/nvcc -o cnn_3d cnn_3d.cpp -O2 -I/usr/local/cuda/include/ -I/usr/include/opencv4/ -L/usr/local/cuda/lib64/ -lcudnn -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
clean:
	rm cnn
	rm cnn_3d
