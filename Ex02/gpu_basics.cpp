/*!
 * \brief Basic code for GPU handling
 * \author Josef Schuele
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "gpu_helper.h"

struct GPUDeviceProp {
 unsigned long GlobalMem, sharedMemPerBlock, maxGridSize;
 int cores, minor, major, multiProc, warp, maxThreads, maxThreadsPerBlock;
 int overlap, concurrent, pagedMemory, eccMemory, unifiedMemory, computeMode;
 float peakPerf, peakmemBW;
 int nstreams; // max number of streams per GPU
 char name[256];
};
int count,initialized=0,selected=-1;

GPUDeviceProp *GPU_Properties;
GPUDeviceProp *GPU_Property;

int GPU_initialize() {
    int major = 0, minor = 0;
    int driverversion=0, runtimeversion=0;
    int dev;
    char deviceName[256],msg[256],*env_;
    cudaDeviceProp deviceProp;
    GPUDeviceProp  *dP;

    if(initialized) return(0);
    initialized = 1;

    cudaError_t error_id = cudaGetDeviceCount(&count);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", 
              (int)error_id, cudaGetErrorString(error_id));
        return(EXIT_FAILURE);
    }
    if (count==0) {
        printf("There is no device that supports CUDA\n");
        return (count);
    }

    GPU_Properties = new GPUDeviceProp [count];

    for (dev = 0; dev < count; ++dev) {
       cudaSetDevice(dev);
       cudaGetDeviceProperties(&deviceProp, dev);
       dP = &GPU_Properties[dev];

       dP->nstreams=0; // initialize max #streams

       //printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
       strcpy(dP->name,deviceProp.name);
      dP->GlobalMem = deviceProp.totalGlobalMem;
      dP->multiProc = deviceProp.multiProcessorCount;
      dP->major = deviceProp.major;
      dP->minor = deviceProp.minor;
      dP->cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      dP->sharedMemPerBlock = deviceProp.sharedMemPerBlock;
      dP->warp = deviceProp.warpSize;
      dP->maxThreads = deviceProp.maxThreadsPerMultiProcessor;
      dP->maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
      dP->overlap = deviceProp.asyncEngineCount;
      dP->concurrent = deviceProp.concurrentKernels;
      dP->pagedMemory = deviceProp.canMapHostMemory;
      dP->eccMemory = deviceProp.ECCEnabled;
      dP->unifiedMemory = deviceProp.unifiedAddressing;
      dP->computeMode = deviceProp.computeMode;
      // dP->peakPerf = (float)deviceProp.clockRate*2.* dP->cores * dP->multiProc * 1.e-6;
      dP->peakPerf = (float)deviceProp.clockRate*2.* dP->cores * dP->multiProc / (1000.0 * 1000.0);
      dP->peakmemBW = (float)deviceProp.memoryClockRate*deviceProp.memoryBusWidth*0.25 / (1000.0 * 1000.0);
      dP->maxGridSize = deviceProp.maxGridSize[0];

      cudaDriverGetVersion(&driverversion);
      cudaRuntimeGetVersion(&runtimeversion);
      printf("  cuda driver version / runtime version          %d.%d / %d.%d\n", 
              driverversion/1000, (driverversion%100)/10, runtimeversion/1000, 
              (runtimeversion%100)/10);
      printf("  cuda capability major/minor version number:    %d.%d\n", 
              deviceProp.major, deviceProp.minor);
      sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
	      (float)deviceProp.totalGlobalMem/1048576.0f, 
              (unsigned long long) deviceProp.totalGlobalMem);
      printf("%s", msg);
      printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
             deviceProp.multiProcessorCount,
             _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
             _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * 
             deviceProp.multiProcessorCount);
      printf("  GPU Max Clock rate:                            %.2f KHz (%0.2f MHz, %0.2f GHz)\n", (float)deviceProp.clockRate, (float)deviceProp.clockRate / 1000.0, (float)deviceProp.clockRate / (1000.0 * 1000.0));
#if CUDART_VERSION >= 5000
      printf("  Memory Clock rate:                             %.2f Khz (%0.2f Ghz)\n", (float)deviceProp.memoryClockRate, (float)deviceProp.memoryClockRate / (1000.0 * 1000.0));
      printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

      if (deviceProp.l2CacheSize)
      {
          printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
      }

#else
      printf(" Installation is pretty old. Install a new one to get full support \n");
      return(EXIT_FAILURE);
#endif

      printf("  Peak Memory bandwidth:			 %f GB/sec\n", dP->peakmemBW);
      printf("  Peak performance:			     %f GHz\n", dP->peakPerf);
      
      printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
      printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      printf("  Warp size:                                     %d\n", deviceProp.warpSize);
      printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
      printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
	      deviceProp.maxThreadsDim[0],
	      deviceProp.maxThreadsDim[1],
	      deviceProp.maxThreadsDim[2]);
      printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
	      deviceProp.maxGridSize[0],
	      deviceProp.maxGridSize[1],
	      deviceProp.maxGridSize[2]);
      printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
      printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
      printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
      printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
      printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
      printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
      printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
      const char *sComputeMode[] =
      {
	  "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
	  "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
	  "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
	  "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
	  "Unknown",
	  NULL
      };
      printf("  Compute Mode:\n");
      printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }
}

void GPU_check_overlap() {
 if(!GPU_Properties[selected].overlap) {
    printf("Selected device can not overlap streams - GPU not usable\n");
    exit(EXIT_FAILURE);
 }
}


/* Set device according to the number of cores */
void GPU_select_cores(void) {
 int i,c,cm = 0;
 if(count==0) return;
 if(!initialized) exit(-1);
 if(selected >= 0) {
  printf("GPU %d is already selected\n",selected);
  return;
 }
 for(i=0;i<count;i++) {
  GPU_Property = &GPU_Properties[i];
  c=GPU_Property->cores*GPU_Property->multiProc;
  printf("Found GPU with %d cores\n",c);
  if(c>cm) { 
     cm=c; 
     selected=i;
  }
 }
 printf("GPU %d , a %s selected\n",selected,GPU_Properties[selected].name);
 GPU_check_overlap();
 cudaSetDevice(selected);
 GPU_Property = &GPU_Properties[selected];
}

/* Set device according to available memory size */
void GPU_select_mem(void) {
 int i;
 unsigned long long m,mm = 0;
 if(count==0) return;
 if(!initialized) exit(-1);
 if(selected >= 0) {
  printf("GPU %d is already selected\n",selected);
  return;
 }
 for(i=0;i<count;i++) {
  GPU_Property = &GPU_Properties[i];
  m=GPU_Property->GlobalMem;
  if(m > mm) {
     mm=m; 
     selected=i;
  }
 }
 GPU_check_overlap();
 cudaSetDevice(selected);
 GPU_Property = &GPU_Properties[selected];
}

void GPU_set_device() {
 if(selected < 0) {
  printf("No device selected yet\n");
  return;
 }
 printf("cudasetDevice %d\n",selected);
 cudaSetDevice(selected);
}

void GPU_show_device() {
 if(!initialized) exit(-1);
 if(selected >= 0) {
  printf("Using GPU %s with Compute Capability %.1d.%.1d\n",
   GPU_Property->name,GPU_Property->major,GPU_Property->minor);
 }
}

/*!
 * \brief Returns true/false about GPU usage
 */
int GPU_usage() { return(count); }

GPUDeviceProp* GPU_get_property() { return(&GPU_Property[selected]); }


int main(){
GPU_initialize();
printf("Initialized\n");
GPU_select_cores();
GPU_set_device();
GPU_show_device();
return 0;
}
