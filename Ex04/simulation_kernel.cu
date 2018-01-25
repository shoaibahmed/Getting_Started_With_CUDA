/* 
References:
http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
*/

#include <stdio.h>
#include <assert.h>

#include "curand.h"
#include "curand_kernel.h"

#define MAX_THREADS_PER_BLOCK 1024
#define USE_PARTICLE_FORCES true

#define ACCELERATION_X 0.0
#define ACCELERATION_Y (-0.981)
#define CONSTANT 3
#define NUM_PARTICLES 200

float* d_particlePosition = NULL;
float* d_particleVelocity = NULL;
float* d_particleMass = NULL;
float* d_particleAcceleration = NULL;
curandState_t* states;

float lastElapsedTime = 0.0;

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

__global__ void ForceComputationKernel(float* particlePosition, float* particleVelocity, float* particleMass, float* particleAcceleration, int DIMX, int DIMY)
{
	int offset = (2 * blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Convert particle position into pixel coordinates
	int screenX = (int) ((particlePosition[offset]) * DIMX);
	int screenY = (int) ((particlePosition[offset+1]) * DIMY);
	if (screenX > 0)
	  screenX--;
	if (screenY > 0)
	  screenY--;

	// Compute force exerted by all other partices on the current particle
	float totalForce_x = 0.0;
	float totalForce_y = 0.0;
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		if (i != blockIdx.x)
		{
			int index = i * 2;
			int screenX_new = (int) ((particlePosition[index]) * DIMX);
			int screenY_new = (int) ((particlePosition[index+1]) * DIMY);
			if (screenX_new > 0)
				screenX_new--;
			if (screenY_new > 0)
				screenY_new--;

			float xDist = (screenX_new - screenX);
			float yDist = (screenY_new - screenY);
			// float xDist = (particlePosition[offset] - particlePosition[index]);
			// float yDist = (particlePosition[offset+1] - particlePosition[index+1]);
			float radius = sqrt((xDist * xDist) + (yDist * yDist));
			if (radius < 0.00001)
			{
				// radius=0.00001;
				radius = 1;
			}
			// F_x = (m_1 * v_1x * m_2 * v_2x) / r
			float currentParticleForce_x = (particleMass[i] * particleVelocity[index] * particleMass[blockIdx.x] * particleVelocity[offset]) / radius;
			float currentParticleForce_y = (particleMass[i] * particleVelocity[index+1] * particleMass[blockIdx.x] * particleVelocity[offset+1]) / radius;

			totalForce_x += (currentParticleForce_x * CONSTANT);
			totalForce_y += (currentParticleForce_y * CONSTANT);
		}
	}

	// Convert the force to acceleration
	float acceleration_x = totalForce_x + ACCELERATION_X;
	float acceleration_y = totalForce_y + ACCELERATION_Y;
	if ((isinf(acceleration_x) || isinf(acceleration_y)) || (isnan(acceleration_x) || isnan(acceleration_y)))
	{
	  if (isnan(acceleration_x) || isnan(acceleration_y))
	    printf("Inf encountered\n");
	  printf("Terminating the program\n");
	  asm("trap;");
	}
	//printf("Acceleration: %f %f\n", acceleration_x, acceleration_y);
	particleAcceleration[offset] = acceleration_x;
	particleAcceleration[offset+1] = acceleration_y;
}

__global__ void SimulationKernel(uchar3 *dary, float* particlePosition, float* particleVelocity, float* particleMass, float* particleAcceleration, float deltaT, int DIMX, int DIMY, curandState_t* states)
{
	deltaT = 0.005;
	// Since the array is ordered in WHC format
	int offset = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

	// Convert particle position into pixel coordinates
	int screenX = (int) ((particlePosition[offset]) * DIMX);
	int screenY = (int) ((particlePosition[offset+1]) * DIMY);
	if (screenX > 0)
		screenX--;
	if (screenY > 0)
		screenY--;

	// Clear the current pixel on the screen
	int realOffset = (screenY * DIMX) + screenX;
	dary[realOffset] = make_uchar3(0, 0, 0);

#if USE_PARTICLE_FORCES
	particleVelocity[offset] = particleVelocity[offset] + (particleAcceleration[offset] * deltaT);
	particleVelocity[offset+1] = particleVelocity[offset+1] + (particleAcceleration[offset+1] * deltaT);

#else
	// Update the parameters of the model
	particleVelocity[offset] = particleVelocity[offset] + (ACCELERATION_X * deltaT);
	particleVelocity[offset+1] = particleVelocity[offset+1] + (ACCELERATION_Y * deltaT);
#endif
	// Adjust the velocity in case the velocity exceeds 1
	if ((abs(particleVelocity[offset] * deltaT) > 1.0) || (abs(particleVelocity[offset+1] * deltaT) > 1.0))
	{
	  particleVelocity[offset] = particleVelocity[offset] * 0.1;
	  particleVelocity[offset+1] = particleVelocity[offset+1] * 0.1;
	}

	particlePosition[offset] = particlePosition[offset] + (particleVelocity[offset] * deltaT);
	particlePosition[offset+1] = particlePosition[offset+1] + (particleVelocity[offset+1] * deltaT);

	// Make sure the particle position is within the range [0,1]
	if (particlePosition[offset] > 1.0)
	{
		particlePosition[offset] = 0.0;
		
	}
	else if (particlePosition[offset] < 0.0)
	{
		particlePosition[offset] = 1.0;
	}

	if (particlePosition[offset+1] > 1.0)
	{
		particlePosition[offset+1] = 0.0;
		particleVelocity[offset] = curand_uniform(&states[blockIdx.x * blockDim.x]); // Thread ID should be replaced with 0 as the value of x is being generated
		particleVelocity[offset+1] = -curand_uniform(&states[blockIdx.x * blockDim.x + 1]); // Thread ID should be replaced with 1 as the value of y is being generated
	}
	else if (particlePosition[offset+1] < 0.0)
	{
		particlePosition[offset+1] = 1.0;
		particleVelocity[offset] = curand_uniform(&states[blockIdx.x * blockDim.x]); // Thread ID should be replaced with 0 as the value of x is being generated
		particleVelocity[offset+1] = -curand_uniform(&states[blockIdx.x * blockDim.x + 1]); // Thread ID should be replaced with 1 as the value of y is being generated
	}

	// Compute the new location of the pixel
	screenX = (int) ((particlePosition[offset]) * DIMX);
	screenY = (int) ((particlePosition[offset+1]) * DIMY);
	if (screenX > 0)
		screenX--;
	if (screenY > 0)
		screenY--;

	// // Add the corresponding color to the pixel
	uchar3 color = make_uchar3(255, (int) (255 * particleVelocity[offset]), (int) (128 * particleVelocity[offset+1]));
	// realOffset = (screenX * DIMY) + screenY;
	realOffset = (screenY * DIMX) + screenX;
	dary[realOffset] = color;
}

void simulate(uchar3 *ptr, int tick, int w, int h)
{
	/* ptr is a pointer to an array of size w*h*sizeof(uchar3).
	   uchar3 is a structure with x,y,z coordinates to contain
	   red,yellow,blue - values for a pixel (Range [0,255])
	*/
	/*
	 [(h-1)w, (h-1)(w+1), (h-1)(w+2), ..., (h-1)(w-1)]
	 ....
	 [w, w+1, w+2, ..., 2w-1]
	 [0, 1, 2, 3, ..., w-1]
	*/
	cudaError_t err=cudaSuccess;
	cudaEvent_t start,stop;
	float elapsedtime;
	bool dataInitialized = false;

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

		err = cudaMalloc(&d_particleMass, sizeof(float) * NUM_PARTICLES);
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle mass.");
			exit(-1);
		}

		err = cudaMalloc(&d_particleAcceleration, sizeof(float) * (NUM_PARTICLES * 2));
		if (err != cudaSuccess)
		{
			printf("Error: Failed to allocate memory for particle acceleration.");
			exit(-1);
		}

		/* allocate space on the GPU for the random states */
		int numParticleComponents = NUM_PARTICLES * 2;
  		cudaMalloc((void**) &states, numParticleComponents * sizeof(curandState_t));

		/* invoke the GPU to initialize all of the random states */
		init<<<NUM_PARTICLES, 2>>>(time(0), states);

		/* invoke the kernel to get some random numbers */
		randoms<<<NUM_PARTICLES, 2>>>(states, d_particlePosition);
		randoms<<<NUM_PARTICLES, 2>>>(states, d_particleVelocity);
		randoms<<<NUM_PARTICLES, 1>>>(states, d_particleMass);

		dataInitialized = true;
	}

	cudaEventCreate  ( &start);
	cudaEventCreate  ( &stop);

	cudaEventRecord(start);

	if (dataInitialized)
		InitializeKernel<<<w, h>>>(ptr);
#if USE_PARTICLE_FORCES
	ForceComputationKernel<<<NUM_PARTICLES, 1>>>(d_particlePosition, d_particleVelocity, d_particleMass, d_particleAcceleration, w, h);
#endif

	SimulationKernel<<<NUM_PARTICLES, 1>>>(ptr, d_particlePosition, d_particleVelocity, d_particleMass, d_particleAcceleration, lastElapsedTime, w, h, states);

	err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime, start, stop);
	printf("Time used: %.1f (ms)\n", elapsedtime);
	lastElapsedTime = elapsedtime;

	cudaEventDestroy  ( start);
	cudaEventDestroy  ( stop);

	printf("Please type ESC in graphics and afterwards RETURN in cmd-screen to finish\n");
}
