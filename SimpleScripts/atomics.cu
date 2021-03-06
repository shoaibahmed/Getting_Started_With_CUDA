#include <stdio.h>
#include <cuda.h>

#define NUM_THREADS 1000000
#define BLOCK_DIM 1000
#define ARRAY_SIZE 10

#define USE_ATOMICS true

__global__ void naiveAddKernel(float* d_arr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx = idx % ARRAY_SIZE;
	d_arr[idx] += 1;
}

__global__ void atomicAddKernel(float* d_arr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx = idx % ARRAY_SIZE;
	atomicAdd(&d_arr[idx], 1);
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
	#if USE_ATOMICS
		atomicAddKernel<<<NUM_THREADS / BLOCK_DIM, BLOCK_DIM>>>(d_arr);
	#else
		naiveAddKernel<<<NUM_THREADS / BLOCK_DIM, BLOCK_DIM>>>(d_arr);
	#endif
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel execution time: %f ms\n", milliseconds);

	// Copy back the results
	cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f", h_arr[i]);
		printf ((i % 4 == 3)? "\n" : "\t");
	}

	cudaFree(h_arr);
	cudaFree(d_arr);

	return 0;
}