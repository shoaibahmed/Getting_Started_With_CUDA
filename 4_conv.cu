#include <stdio.h>
#include <math.h>

// Kernel addition on GPU
__global__ void conv (float** I, float** F, int H, int W, int K, int filterPad, float** out)
{
	int m = threadIdx.x;
	int n = threadIdx.y;

	float convOut = 0.0f;
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			// For correlation
			int indI = m - filterPad + i;
			int indJ = n - filterPad + j;

			// For convolution
			// int indI = m + filterPad - i;
			// int indJ = n + filterPad - j;

			if (indI < 0 || indI >= H || indJ < 0 || indJ >= W)
			{

			}
			else
			{
				convOut += I[indI][indJ] * F[i][j]; // Correlation
				// convOut += I[m + filterPad - i][n + filterPad - j] * F[i][j]; // Convolution
			}
		}
	}
	out[m][n] = convOut;
}

void init (float** I, float** F, int H, int W, int K, int filterPad)
{
	// Initialize I
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			I[i][j] = (i * W) + (j);	
		}
	}

	// Initialize F
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < K; j++)
		{
			F[i][j] = j - filterPad;
		}
	}
}

// Main function on the host
int main()
{
	int H = 5, W = 5, K = 3;
	int filterPad = (int) (floor(K) / 2.0f);

	float **I, **dev_I;
	float **F, **dev_F;
	float **dev_Out;

	printf ("Allocating memory on Host\n");
	I = (float **) malloc(H * sizeof(float *));
	for (int i = 0; i < H; i++)
		I[i] = (float*) malloc(sizeof(float) * W);

	F = (float **) malloc(K * sizeof(float *));
	for (int i = 0; i < K; i++)
		F[i] = (float*) malloc(sizeof(float) * K);
	printf ("Host memory allocated\n");

	printf ("Initializing array\n");
	init(I, F, H, W, K, filterPad); // Initialize the array
	printf ("Initialization complete\n");

	printf ("Allocating device memory\n");
	cudaMalloc((void **) &dev_I, sizeof(float *) * H);
	for (int i = 0; i < H; i++)
		cudaMalloc((void **) &dev_I[i], sizeof(float) * W);

	cudaMalloc((void **) &dev_F, sizeof(float *) * K);
	for (int i = 0; i < K; i++)
		cudaMalloc((void **) &dev_F[i], sizeof(float) * K);

	cudaMalloc((void **) &dev_Out, sizeof(float *) * H);
	for (int i = 0; i < H; i++)
		cudaMalloc((void **) &dev_Out[i], sizeof(float) * W);
	printf ("Device memory allocated\n");

	printf ("Moving data to device\n");
	for (int i = 0; i < H; i++)
		cudaMemcpy(dev_I[i], I[i], sizeof(float) * W, cudaMemcpyHostToDevice);
	for (int i = 0; i < K; i++)
		cudaMemcpy(dev_F[i], F[i], sizeof(float) * K, cudaMemcpyHostToDevice);
	printf ("Data moved to device\n");

	printf ("Performing convolution\n");
	conv <<< H, W >>> (dev_I, dev_F, H, W, K, filterPad, dev_Out);
	printf ("Kernel sucessfully executed!\n");

	printf ("Moving data back to host\n");
	for (int i = 0; i < H; i++)
		cudaMemcpy(I[i], dev_Out[i], sizeof(float) * W, cudaMemcpyDeviceToHost);
	printf ("Data moved to host\n");

	cudaFree(dev_I);
	cudaFree(dev_F);
	cudaFree(dev_Out);
	printf ("Device memory released\n");

	print ("Convolution output:\n")
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf ("%.2f\t", I[i][j]);
		}
		printf ("\n");
	}

	return 0;
}