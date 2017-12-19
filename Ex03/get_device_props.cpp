#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

#include <stdio.h>

int main()
{
  // Query the system for the device properties
  int numberOfCUDADevices;
  cudaGetDeviceCount(&numberOfCUDADevices);

  //Iterate over all the cuda devices
  for (int i = 0; i < numberOfCUDADevices; i++)
  {
    printf("Information from Device # %d\n", i);
    
    // Get the properties
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    
    // Print the information from the device
    printf("\t Device name: %s\n", deviceProps.name);
    printf("\t Total Global Memory (Bytes): %d\n", deviceProps.totalGlobalMem);
    printf("\t Shared Memory per block (Bytes): %d\n", deviceProps.sharedMemPerBlock);
    printf("\t Registers per block: %d\n", deviceProps.regsPerBlock);
    printf("\t Warp size: %d\n", deviceProps.warpSize);
    
    printf("\t Memory pitch size (Bytes): %d\n", deviceProps.warpSize);
    
    printf("\t Memory Clock Rate (KHz): %d\n", deviceProps.memoryClockRate);
    printf("\t Memory Bus Width (bits): %d\n", deviceProps.memoryBusWidth);
  }
  
  return 0;
}
