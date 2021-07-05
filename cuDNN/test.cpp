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
    if (status != CUDNN_STATUS_SUCCESS)                      \
    {                                                        \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char *image_path)
{
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}

int main(int argc, char const *argv[])
{
  // Initialize cudnn
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  // Load image data
  cv::Mat image = load_image("lena.png");
  if (image.empty())
  {
    std::cerr << "Unable to load image..." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "Image size: " << image.cols << ", " << image.rows << std::endl;

  // Initialize cudnn descriptor for the input
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

  // Initialize cudnn descriptor for the output
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

  // Initialize cudnn descriptor for the kernel
  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/3,
                                        /*in_channels=*/3,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

  // Initialize cudnn descriptor for the convolution operation
  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

#if CUDNN_MAJOR >= 8
  // Use the recent interface to get the convolution algorithm
  int requestedAlgoCount = 0, returnedAlgoCount = 0;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requestedAlgoCount));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> results(requestedAlgoCount);
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
      input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor,
      requestedAlgoCount,
      &returnedAlgoCount,
      &results[0]));

  bool found_conv_algorithm = false;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  size_t workspace_bytes = 0;
  size_t free_memory, total_memory;
  for (int i = 0; i < returnedAlgoCount; i++)
  {
      if (results[i].status == CUDNN_STATUS_SUCCESS &&
          results[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
          results[i].memory < free_memory)
      {
          found_conv_algorithm = true;
          convolution_algorithm = results[i].algo;
          workspace_bytes = results[i].memory;
          break;
      }
  }
  assert(found_conv_algorithm);

#else
  // Select the appropriate convolution algorithm
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                 input_descriptor,
                                                 kernel_descriptor,
                                                 convolution_descriptor,
                                                 output_descriptor,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 /*memoryLimitInBytes=*/0,
                                                 &convolution_algorithm));
  
  // Compute memory requirements for the operation
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
#endif
  
  // Allocate the required memory for the algorithm
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
  void *d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int batch_size = 1, channels = 3, height = image.rows, width = image.cols;
  int image_bytes = batch_size * channels * height * width * sizeof(float);

  float *d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

  float *d_output{nullptr};
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);
}
