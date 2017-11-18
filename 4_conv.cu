#include <stdio.h>
#include <math.h>

// Kernel addition on GPU
__global__ void conv (float** I, float** F, int H, int W, int K, int filterPad, float** out)
{
	int m = threadIdx.y; // First dim
	int n = threadIdx.x; // Second dim

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
	float **out, **dev_Out;

	printf ("Allocating memory on Host\n");
	I = (float **) malloc(H * sizeof(float *));
	for (int i = 0; i < H; i++)
		I[i] = (float*) malloc(sizeof(float) * W);

	F = (float **) malloc(K * sizeof(float *));
	for (int i = 0; i < K; i++)
		F[i] = (float*) malloc(sizeof(float) * K);

	out = (float **) malloc(H * sizeof(float *));
	for (int i = 0; i < H; i++)
		out[i] = (float*) malloc(sizeof(float) * W);
	printf ("Host memory allocated\n");

	printf ("Initializing array\n");
	init(I, F, H, W, K, filterPad); // Initialize the array
	printf ("Initialization complete\n");

	printf ("Allocating device memory for I\n");
	cudaError_t err = cudaMalloc((void **) &dev_I, sizeof(float *) * H);
	if(err != cudaSuccess)
	{
		printf("Failure in allocating I\n");
		exit(1);
	}
	for (int i = 0; i < H; i++)
	{
		// cudaMalloc((void **) &dev_I[i], sizeof(float) * W);
		err = cudaMalloc((void **) &I[i], sizeof(float) * W);
		if(err != cudaSuccess)
		{
			printf("Failure in populating I\n");
			exit(1);
		}
	}

	printf ("Allocating device memory for F\n");
	err = cudaMalloc((void **) &dev_F, sizeof(float *) * K);
	if(err != cudaSuccess)
	{
		printf("Failure in allocating F\n");
		exit(1);
	}
	for (int i = 0; i < K; i++)
	{
		err = cudaMalloc((void **) &F[i], sizeof(float) * K);
		if(err != cudaSuccess)
		{
			printf("Failure in populating K\n");
			exit(1);
		}
	}

	printf ("Allocating device memory for output\n");
	err = cudaMalloc((void **) &dev_Out, sizeof(float *) * H);
	if(err != cudaSuccess)
	{
		printf("Failure in allocating output array\n");
		exit(1);
	}
	for (int i = 0; i < H; i++)
	{
		err = cudaMalloc((void **) &out[i], sizeof(float) * W);
		if(err != cudaSuccess)
		{
			printf("Failure in populating ouput array\n");
			exit(1);
		}
	}
	printf ("Device memory allocated\n");

	printf ("Moving data to device\n");
	// for (int i = 0; i < H; i++)
	// 	cudaMemcpy(dev_I[i], I[i], sizeof(float) * W, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_I, I, sizeof(float *) * H, cudaMemcpyHostToDevice);
	// for (int i = 0; i < K; i++)
	// 	cudaMemcpy(dev_F[i], F[i], sizeof(float) * K, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_F, F, sizeof(float *) * K, cudaMemcpyHostToDevice);
	printf ("Data moved to device\n");

	printf ("Performing convolution\n");
	conv <<< H, W >>> (dev_I, dev_F, H, W, K, filterPad, dev_Out);
	printf ("Kernel sucessfully executed!\n");

	printf ("Moving data back to host\n");
	// for (int i = 0; i < H; i++)
	// 	cudaMemcpy(I[i], dev_Out[i], sizeof(float) * W, cudaMemcpyDeviceToHost);
	cudaMemcpy(out, dev_Out, sizeof(float *) * H, cudaMemcpyDeviceToHost);
	printf ("Data moved to host\n");

	cudaFree(dev_I);
	cudaFree(dev_F);
	cudaFree(dev_Out);
	printf ("Device memory released\n");

	printf ("Convolution output:\n");
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf ("%.2f\t", out[i][j]);
		}
		printf ("\n");
	}

	return 0;
}