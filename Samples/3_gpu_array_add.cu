#include "stdio.h"

// Kernel addition on GPU
__global__ void add (int N, float a, float b, float c, float* A)
{
	int i = threadIdx.x;
	A[i] = (A[i] + a) * b + c;
}

void init (float* A, int N)
{
	for (int i = 0; i < N; i++)
	{
		A[i] = (float)i;
		// printf ("%d: %.2f\n", i, A[i]);
	}
}

// Main function on the host
int main()
{
	int N = 20;
	float *A, *dev_A;
	A = (float*) malloc(sizeof(float) * N);
	printf ("Initializing array\n");
	init(A, N); // Initialize the array
	printf ("Initialization complete\n");
	cudaMalloc((void **) &dev_A, sizeof(float) * N);
	printf ("Device memory allocated\n");
	cudaMemcpy(dev_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
	printf ("Data moved to device\n");
	add <<< 1, N >>> (N, 3.0f, 4.0f, -2.0f, dev_A);
	cudaMemcpy(A, dev_A, sizeof(float) * N, cudaMemcpyDeviceToHost);
	printf ("Data moved to host\n");
	cudaFree(dev_A);
	printf ("Device memory released\n");

	for (int i = 0; i < N; i++)
	{
		printf ("%.2f ", A[i]);
	}
	printf ("\n");

	return 0;
}