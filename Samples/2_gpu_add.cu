#include "stdio.h"

// Kernel addition on GPU
__global__ void add(int a, int* b)
{
	*b += a * 100;
}

// Main function on the host
int main()
{
	int b, *dev_b;
	cudaMalloc((void **) &dev_b, sizeof(int));
	b = 4;
	cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	add <<< 1, 1 >>> (2, dev_b);
	cudaMemcpy(&b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_b);
	printf ("B: %d\n", b);
	return 0;
}