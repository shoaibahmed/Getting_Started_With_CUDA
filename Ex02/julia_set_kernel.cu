#include <stdio.h>
#include <assert.h>
// #include <math.h>

#define MAX_THREADS_PER_BLOCK 1024
#define NUM_ITERATIONS 300

#define CONSTANT_1 -0.7885
#define CONSTANT_2 0.7885

//__global__ void JuliaSetKernel(uchar3 *dary, float* real, float* imaginary, int t, int DIMX, int DIMY, int numBlocksWithSameColor)
__global__ void JuliaSetKernel(uchar3 *dary, int t, int DIMX, int DIMY)
{
	/* Insert your kernel here */
	// Since the array is ordered in WHC format
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Initialize the value of z using the initial c
	float z_realPart = ((threadIdx.x / ((float)blockDim.x)) * 2.0) - 1.0;
	float z_imaginaryPart = ((blockIdx.x / ((float)gridDim.x)) * 2.0) - 1.0;
	// printf ("Real: %f | Imaginary: %f | Offset: %d\n", z_realPart, z_imaginaryPart, offset);

	// Add C in Zn
	// z_realPart = z_realPart - 0.8;
	// z_imaginaryPart = z_imaginaryPart + 0.156;

	// Perform 300 iterations
	// float A;
	// if ((t / 100) % 2 == 0)
	// 	A = 0.08 + (t % 100) / 1220.0;
	// else
	// 	A = 0.08 + (99 - (t % 100)) / 1220.0;
	float A = 0.08 + (t % 100) / 1220.0;

	// Euler formula (cos(A) + i sin(A))
	float eulerReal = CONSTANT_1 * cos(A);
	float eulerImaginary = CONSTANT_2 * sin(A);
	// printf ("Euler Real: %f | Euler Imaginary: %f\n", eulerReal, eulerImaginary);

	for (int time = 0; time < NUM_ITERATIONS; time++)
	{
		// Update Zn based on the new values
		// Complex multiplication: (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
		float z_realPart_new = ((z_realPart * z_realPart) - (z_imaginaryPart * z_imaginaryPart)) + eulerReal;
		float z_imaginaryPart_new = ((z_realPart * z_imaginaryPart) + (z_realPart * z_imaginaryPart)) + eulerImaginary;

		// Update the values
		z_realPart = z_realPart_new;
		z_imaginaryPart = z_imaginaryPart_new;
		// printf ("Real: %f | Imaginary: %f | Offset: %d\n", z_realPart, z_imaginaryPart, offset);
	}
	
	// Determine if the value is different from others	
	float zNormSquared = pow(z_realPart, 2) + pow(z_imaginaryPart, 2);

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

	JuliaSetKernel<<<w, h>>>(ptr, tick, w, h);
	
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
