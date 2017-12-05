#include <stdio.h>
#include <assert.h>

#define MAX_THREADS_PER_BLOCK 1024
#define NUM_ITERATIONS 300

#define ANIMATE_SET true

#if ANIMATE_SET == true
#define CONSTANT_1 -0.7885
#define CONSTANT_2 0.7885
#else
#define CONSTANT_1 -0.8
#define CONSTANT_2 0.156
#endif

#define SCALE 2.0
#define SHIFT 1.0

__global__ void JuliaSetKernel(uchar3 *dary, int t, int DIMX, int DIMY)
{
	/* Insert your kernel here */
	// Since the array is ordered in WHC format
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Initialize the value of z using the initial c
	float z_realPart = ((threadIdx.x / ((float)blockDim.x)) * SCALE) - SHIFT;
	float z_imaginaryPart = ((blockIdx.x / ((float)gridDim.x)) * SCALE) - SHIFT;

#if ANIMATE_SET == true
	float A = 0.08 + ((t % 100) / 1220.0);
	
	// Euler formula e^iA = (cos(A) + i sin(A))
	float eulerReal = CONSTANT_1 * cos(A);
	float eulerImaginary = CONSTANT_2 * sin(A);
#else
	float eulerReal = CONSTANT_1;
	float eulerImaginary = CONSTANT_2;
#endif

	for (int time = 0; time < NUM_ITERATIONS; time++)
	{
		// Update Zn based on the new values
		// Complex multiplication: (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
		float z_realPart_new = ((z_realPart * z_realPart) - (z_imaginaryPart * z_imaginaryPart)) + eulerReal;
		float z_imaginaryPart_new = ((z_realPart * z_imaginaryPart) + (z_realPart * z_imaginaryPart)) + eulerImaginary;
		
		// Update the values
		z_realPart = z_realPart_new;
		z_imaginaryPart = z_imaginaryPart_new;
	}

	// Determine if the value is different from others	
	//float zNormSquared = pow(z_realPart, 2) + pow(z_imaginaryPart, 2);
	float zNormSquared = (z_realPart * z_realPart) + (z_imaginaryPart * z_imaginaryPart);

	// Add the corresponding color to the pixel
	uchar3 color;
	if (zNormSquared < 2.0)
		color = make_uchar3(255, 0, 0);
	else
		color = make_uchar3(0, 0, 0);

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

	JuliaSetKernel<<<w, h>>>(ptr, tick, h, w);
	
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
