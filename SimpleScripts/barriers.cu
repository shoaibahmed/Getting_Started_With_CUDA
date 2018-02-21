#include <stdio.h>
#include <cuda.h>

#define NUM_THREADS 100
#define BLOCK_DIM 100
#define ARRAY_SIZE 100

__global__ void barriersTestKernel(float* d_arr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_arr[idx] += 1;
	// __syncthreads();
	float sum = 0;
	for (int i = 0; i <= threadIdx.x; i++)
	{
		sum += d_arr[i];
	}
	// __syncthreads();
	d_arr[idx] = sum;
	sum = 0.0;
	for (int i = 0; i <= threadIdx.x; i++)
	{
		sum += d_arr[i];
	}
	d_arr[idx] = sum;
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Initialize the array
	float *d_arr, *h_arr;
	cudaMalloc((void **) &d_arr, ARRAY_SIZE * sizeof(float));
	cudaMemset((void **) &d_arr, 0, sizeof(float));

	cudaEventRecord(start);
	barriersTestKernel<<<NUM_THREADS / BLOCK_DIM, BLOCK_DIM>>>(d_arr);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel execution time: %f ms\n", milliseconds);

	// Copy back the results
	cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f", h_arr[i]);
		printf((i % 4 == 3)? "\n" : "\t");
	}

	cudaFree(h_arr);
	cudaFree(d_arr);

	return 0;
}