#include <stdio.h>
#include <cuda.h>

#define ARRAY_SIZE 64

__global__ void SquareKernel(float *d_out, float *d_in)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float val = d_in[idx];
	d_out[idx] = val * val; 
}

int main()
{
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// Allocate the array on the host
	float *h_in, *h_out;
	h_in = (float *) malloc(ARRAY_BYTES);
	h_out = (float *) malloc(ARRAY_BYTES);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = (float) i;
	}

	// Allocate arrays onto the device
	float *d_in, *d_out;
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Launch the kernel
	dim3 gridDim(ARRAY_SIZE / 2, 1, 1);
	dim3 blockDim(ARRAY_SIZE / 2, 1, 1);
	SquareKernel<<<gridDim, blockDim>>> (d_out, d_in);

	// Copy back the results
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// Print the results
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf ("%f", h_out[i]);
		printf ((i % 4 == 3)? "\n" : "\t");
	}

	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}