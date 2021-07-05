/*
 References:
    http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
    https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate
 */

#include <iostream>
#include <cudnn.h>
#include <opencv2/opencv.hpp>


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

int main(int argc, char const *argv[]) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cv::Mat image = load_image("image.png");
    if(image.empty()) {
        std::cerr << "Unable to load image..." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
