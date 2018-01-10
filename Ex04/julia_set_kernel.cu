#include <stdio.h>
#include <assert.h>

#include "curand.h"

#define MAX_THREADS_PER_BLOCK 1024
//#define NUM_ELEMENTS (800 * 800 * 2)
#define NUM_ELEMENTS (800 * 2)

float* d_particlePosition = NULL;
float* d_particleVelocity = NULL;

#define ACCELERATION -9.81

__global__ void SimulationKernel(uchar3 *dary, float* particlePosition, float* particleVelocity, int t, int DIMX, int DIMY)
{
	// Since the array is ordered in WHC format
	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	// Assign the colors only to the corresponding threads with a partcile position
	int posX = 0;
	int posY = 0;
	int withinRange = 0;
	if ((offset * 2) < NUM_ELEMENTS)
	{
		withinRange = 1;
		posX = (int)(DIMX * particlePosition[offset*2]);
		posY = (int)(DIMY * particlePosition[(offset*2)+1]);
	}

	// Add the corresponding color to the pixel
	uchar3 color;
	if (withinRange)
	{
		color = make_uchar3(255, 0, 0);
	}
	else
	{
		color = make_uchar3(0, 0, 0);
	}
	int realOffset = (posX * DIMX) + posY;
	dary[realOffset] = color;
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

	// Initialize the data structures if not yet initialized
	if (d_particlePosition == NULL)
	{
		err = cudaMalloc(&d_particlePosition, sizeof(float) * NUM_ELEMENTS);
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle position.");
			exit(-1);
		}

		err = cudaMalloc(&d_particleVelocity, sizeof(float) * NUM_ELEMENTS);
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle velocity.");
			exit(-1);
		}

		//allocate space for 100 floats on the GPU
		//could also do this with thrust vectors and pass a raw pointer
		curandGenerator_t gen;
		srand(time(NULL));
		int _seed = rand();
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, _seed);
		curandGenerateUniform(gen, d_particlePosition, NUM_ELEMENTS); //generate the random numbers
		curandGenerateUniform(gen, d_particleVelocity, NUM_ELEMENTS); //generate the random numbers
		curandDestroyGenerator(gen);
	}

	cudaEventCreate  ( &start);
	cudaEventCreate  ( &stop);

	cudaEventRecord(start);

	SimulationKernel<<<w, h>>>(ptr, d_particlePosition, d_particleVelocity, tick, h, w);

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
