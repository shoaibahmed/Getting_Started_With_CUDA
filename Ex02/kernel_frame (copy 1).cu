#include <stdio.h>
#include <assert.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 1024

__global__ void Yourkernel(uchar3 *dary,float t,int DIMX,int DIMY, int numBlocksWithSameColor)
{
	/* Insert your kernel here */
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (i * DIMX) + (j);
	
	uchar3 color;
	// color = make_uchar3(((float)i / DIMX) * 256, ((float)j / DIMY) * 256 , 0);
	
	// Distinct color for each block
	// color = make_uchar3(((float)i / DIMX) * 256, ((float)j / DIMY) * 256, 0);
	// color = make_uchar3(((float)blockIdx.x / gridDim.x) * 255, ((float)blockIdx.y / gridDim.y) * 255, 0);
	int blockColorIdxX = blockIdx.x / numBlocksWithSameColor;
	int normalizerX = gridDim.x / numBlocksWithSameColor;
	int blockColorIdxY = blockIdx.y / numBlocksWithSameColor;
	int normalizerY = gridDim.y / numBlocksWithSameColor;
	color = make_uchar3(((float)blockColorIdxX / normalizerX) * 255, ((float)blockColorIdxY / normalizerY) * 255, 0);
	dary[offset] = color;
}

void simulate(uchar3 *ptr, int tick, int w, int h)
{
	/* ptr is a pointer to an array of size w*h*sizeof(uchar3).
	   uchar3 is a structure with x,y,z coordinates to contain
	   red,yellow,blue - values for a pixel (Range [0,255])
	*/
	cudaError_t err=cudaSuccess;
	cudaEvent_t start,stop;
	float elapsedtime;

	cudaEventCreate  ( &start);
	cudaEventCreate  ( &stop);

	cudaEventRecord(start);

	/* Space for
	Yourkernel
	*/
	int divisions = 3; // 9 blocks
	int blockDim = 25;

	// Pick the ideal dimensions of kernel

	dim3 dimBlock(blockDim, blockDim);
	// dim3 dimBlock((int)(w / divisions), (int)(h / divisions));
	dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
	printf("Grid dims: (%d, %d)\n", dimGrid.x, dimGrid.y);

	// Determine the number of kernels to be colored the same
	int numBlocksWithSameColor = floor(h / (divisions * blockDim));
	printf("Number of blocks with same color: %d\n", numBlocksWithSameColor);
	
	// Start the kernel
	Yourkernel<<<dimGrid, dimBlock>>>(ptr, tick, w, h, numBlocksWithSameColor);

	err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime, start, stop);
	printf("Time used: %.1f (ms)\n",elapsedtime);

	cudaEventDestroy  ( start);
	cudaEventDestroy  ( stop);

	printf("Please type ESC in graphics and afterwards RETURN in cmd-screen to finish\n");
}
