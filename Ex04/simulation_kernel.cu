#include <stdio.h>
#include <assert.h>

#include "curand.h"
#include "curand_kernel.h"

#define MAX_THREADS_PER_BLOCK 1024
#define ACCELERATION -9.81
#define NUM_PARTICLES 200

float* d_particlePosition = NULL;
float* d_particleVelocity = NULL;

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				blockIdx.x * blockDim.x + threadIdx.x , /* the sequence number should be different for each core (unless you want all
							 cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[blockIdx.x * blockDim.x + threadIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, float* numbers) {
	/* curand works like rand - except that it takes a state as a parameter */
	// numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
	numbers[blockIdx.x * blockDim.x + threadIdx.x] = curand_uniform(&states[blockIdx.x * blockDim.x + threadIdx.x]);
}

__global__ void InitializeKernel(uchar3 *dary)
{
	// Since the array is ordered in WHC format
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	uchar3 color;
	color = make_uchar3(0, 0, 0);
	dary[offset] = color;
}

__global__ void SimulationKernel(uchar3 *dary, float* particlePosition, float* particleVelocity, int t, int DIMX, int DIMY)
{
	// Since the array is ordered in WHC format
	int offset = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	// Convert particle position into pixel coordinates
	int screenX = (int) (particlePosition[offset] * DIMX);
	int screenY = (int) (particlePosition[offset+1] * DIMY);

	// // Add the corresponding color to the pixel
	uchar3 color = make_uchar3(255, 0, 0);
	int realOffset = (screenX * DIMX) + screenY;
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
	// bool dataInitialized = false;

	// Initialize the data structures if not yet initialized
	if (d_particlePosition == NULL)
	{
		// Allocate memory equivalent to NUM_PARTICLES * 2 as the position and velocity is comprised of a x as well as a y component
		err = cudaMalloc(&d_particlePosition, sizeof(float) * (NUM_PARTICLES * 2));
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle position.");
			exit(-1);
		}

		err = cudaMalloc(&d_particleVelocity, sizeof(float) * (NUM_PARTICLES * 2));
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle velocity.");
			exit(-1);
		}

		/* CUDA's random number library uses curandState_t to keep track of the seed value
		we will store a random state for every thread  */
		curandState_t* states;

		/* allocate space on the GPU for the random states */
		int numParticleComponents = NUM_PARTICLES * 2;
  		cudaMalloc((void**) &states, numParticleComponents * sizeof(curandState_t));

		/* invoke the GPU to initialize all of the random states */
		init<<<numParticleComponents, numParticleComponents>>>(time(0), states);

		/* invoke the kernel to get some random numbers */
		randoms<<<numParticleComponents, numParticleComponents>>>(states, d_particlePosition);
		randoms<<<numParticleComponents, numParticleComponents>>>(states, d_particleVelocity);
		cudaFree(states);

		// dataInitialized = true;
	}

	cudaEventCreate  ( &start);
	cudaEventCreate  ( &stop);

	cudaEventRecord(start);

	// if (dataInitialized)
	// 	InitializeKernel<<<w, h>>>(ptr);

	SimulationKernel<<<NUM_PARTICLES, 1>>>(ptr, d_particlePosition, d_particleVelocity, tick, h, w);

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
