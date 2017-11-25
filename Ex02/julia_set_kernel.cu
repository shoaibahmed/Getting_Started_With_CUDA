#include <stdio.h>
#include <assert.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_VAL 1.0
#define NUM_VALUES 1000

typedef struct __align__(8) { 
    float real;
    float img;
} complex;

__global__ complex operator+(const complex &a, const complex &b)
{
  complex result;
  result.real = a.real + b.real;
  result.img = a.img + b.img;

  return result;
}

complex* complexData;
/* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
curandState_t* states;

/* allocate space on the GPU for the random states */
cudaMalloc((void**) &states, NUM_VALUES * sizeof(curandState_t));

/* invoke the GPU to initialize all of the random states */
init<<<N, 1>>>(time(0), states);

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__device__ void randoms(curandState_t* states, complex* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
float randomNumber = curand_uniform(&states[blockIdx.x]);
  numbers[blockIdx.x] = randomNumber;
}

__global__ void JuliaSetKernel(uchar3 *dary, complex *complexData, int t, int DIMX, int DIMY, int numBlocksWithSameColor)
{
	/* Insert your kernel here */
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Initialize the values of the complex data in the first iteration
	if (t == 0)
	{

	}

	// Ignore threads outside the canvas range (imperfect division in number of blocks)
	if (i >= DIMX)
		return;
	if (j >= DIMY)
		return;

	int offset = (i * DIMX) + (j);

	uchar3 color;

	int blockColorIdxX = blockIdx.x / numBlocksWithSameColor + 1;
	int normalizerX = gridDim.x / numBlocksWithSameColor + 1;
	float xProportion = (float)((blockIdx.x % numBlocksWithSameColor) * blockDim.x + threadIdx.x) / (numBlocksWithSameColor * blockDim.x);
	int blockColorIdxY = blockIdx.y / numBlocksWithSameColor + 1;
	int normalizerY = gridDim.y / numBlocksWithSameColor + 1;
	float yProportion = (float)((blockIdx.y % numBlocksWithSameColor) * blockDim.y + threadIdx.y) / (numBlocksWithSameColor * blockDim.y);

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

	if (tick == 0)
	{
		// Take 1000 equidistant numbers
		cudaMalloc((void **) &complexData, sizeof(complex) * NUM_VALUES);
	}

	/* Space for
	Yourkernel
	*/
	int divisions = 3; // 9 blocks
	int blockDim = 32;

	// Pick the ideal dimensions of kernel

	dim3 dimBlock(blockDim, blockDim);
	// dim3 dimBlock((int)(w / divisions), (int)(h / divisions));
	dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
	printf("Grid dims: (%d, %d)\n", dimGrid.x, dimGrid.y);

	// Determine the number of kernels to be colored the same
	int numBlocksWithSameColor = floor(h / (divisions * blockDim));
	printf("Number of blocks with same color: %d\n", numBlocksWithSameColor);
	
	// Start the kernel
	JuliaSetKernel<<<dimGrid, dimBlock>>>(ptr, complexData, tick, w, h, numBlocksWithSameColor);

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
