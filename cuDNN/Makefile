all:
	/usr/local/cuda/bin/nvcc -o test test.cpp -O2 -I/usr/local/cuda/include/ -I/usr/include/opencv4/ -L/usr/local/cuda/lib64/ -lcudnn -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	# g++ -o test test.cpp -I/usr/local/cuda/include/ -I/usr/include/opencv4/ -L/usr/local/cuda/lib64/ -lcudnn -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
clean:
	rm test
