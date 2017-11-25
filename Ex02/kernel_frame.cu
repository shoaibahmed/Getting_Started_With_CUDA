#include <stdio.h>
#include <assert.h>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 1024
#define DRAW_GRADIENT_MAP true

__global__ void ColorBufferFillKernel(uchar3 *dary, float t, int DIMX, int DIMY, int numBlocksWithSameColorForH, int numBlocksWithSameColorForW)
{
	/* Insert your kernel here */
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Ignore threads outside the canvas range (imperfect division in number of blocks)
	if (i >= DIMX)
		return;
	if (j >= DIMY)
		return;

	// Since the array is ordered in WHC format
	int offset = (i) + (j * DIMX);

	uchar3 color;
	
#if DRAW_GRADIENT_MAP
	// color = make_uchar3(((float)i / DIMX) * 256, ((float)j / DIMY) * 256 , 0);
	
	// Distinct color for each block
	// color = make_uchar3(((float)i / DIMX) * 256, ((float)j / DIMY) * 256, 0);
	// color = make_uchar3(((float)blockIdx.x / gridDim.x) * 255, ((float)blockIdx.y / gridDim.y) * 255, 0);

	int blockColorIdxX = blockIdx.x / numBlocksWithSameColorForW + 1;
	int normalizerX = gridDim.x / numBlocksWithSameColorForW + 1;
	float xProportion = (float)((blockIdx.x % numBlocksWithSameColorForW) * blockDim.x + threadIdx.x) / (numBlocksWithSameColorForW * blockDim.x);
	int blockColorIdxY = blockIdx.y / numBlocksWithSameColorForH + 1;
	int normalizerY = gridDim.y / numBlocksWithSameColorForH + 1;
	float yProportion = (float)((blockIdx.y % numBlocksWithSameColorForH) * blockDim.y + threadIdx.y) / (numBlocksWithSameColorForH * blockDim.y);

	int currentBlockColorX = (((float)blockColorIdxX / normalizerX) * 255);
	int currentBlockColorY = (((float)blockColorIdxY / normalizerY) * 255);

	// Get last block colors
	int lastBlockXColor = 0, lastBlockYColor = 0;
	if (blockColorIdxX > 0)
		lastBlockXColor = ((float)(blockColorIdxX - 1) / normalizerX) * 255;
	if (blockColorIdxY > 0)
		lastBlockYColor = ((float)(blockColorIdxY - 1) / normalizerX) * 255;

	// color = make_uchar3(((float)blockColorIdxX / normalizerX) * 255, ((float)blockColorIdxY / normalizerY) * 255, 0);
	color = make_uchar3((xProportion) * currentBlockColorX + (1.0 - xProportion) * lastBlockXColor, 
		(yProportion) * currentBlockColorY + (1.0 - yProportion) * lastBlockYColor, 0);

#else
	int blockColorIdxX = blockIdx.x / numBlocksWithSameColorForW;
	int normalizerX = gridDim.x / numBlocksWithSameColorForW;
	int blockColorIdxY = blockIdx.y / numBlocksWithSameColorForH;
	int normalizerY = gridDim.y / numBlocksWithSameColorForH;
	color = make_uchar3(((float)blockColorIdxX / normalizerX) * 255, ((float)blockColorIdxY / normalizerY) * 255, 0);
	
#endif

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
	int numBlocksWithSameColorForH = floor(h / (divisions * blockDim));
	int numBlocksWithSameColorForW = floor(w / (divisions * blockDim));
	printf("Number of blocks with same color for H: %d and for W: %d\n", numBlocksWithSameColorForH, numBlocksWithSameColorForW);
	
	// Start the kernel
	ColorBufferFillKernel<<<dimGrid, dimBlock>>>(ptr, tick, w, h, numBlocksWithSameColorForH, numBlocksWithSameColorForW);

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
