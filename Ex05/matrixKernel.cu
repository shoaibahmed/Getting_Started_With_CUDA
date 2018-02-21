#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>

#define MAX_BLOCKS 8
#define MAX_THREADS 1024

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void cublas_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasStatus_t ret = cublasCreate(&handle);
  if (ret != CUBLAS_STATUS_SUCCESS)
  {
    printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

void printMatrix(float *mat, int nRows, int nCols)
{
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			printf("%f\t", mat[i * nCols + j]);
		}
		printf("\n");
	}
}

int main()
{
	cudaError_t c_e;
	cudaEvent_t start_kernel, stop_kernel;
	float kernel_time;

	// Initilize the matrix
	int nRows_A = 3, nRows_B = 3;
	int nCols_A = 3, nCols_B = 3;
  assert(nCols_A == nRows_B);

	float *A, *B, *C;
	A = (float*) malloc(nRows_A * nCols_A * sizeof(float));
	B = (float*) malloc(nRows_B * nCols_B * sizeof(float));
	C = (float*) malloc(nRows_A * nCols_B * sizeof(float));

	// Copy data to the matrix
	for (int i = 0; i < nRows_A; i++)
	{
		for (int j = 0; j < nCols_A; j++)
		{
			A[i * nCols_A + j] = i * nCols_A + j;
		}
	}
	for (int i = 0; i < nRows_B; i++)
	{
		for (int j = 0; j < nCols_B; j++)
		{
			B[i * nCols_B + j] = (nRows_B * nCols_B) - (i * nCols_A + j);
		}
	}
	printf("Matrix A\n");
	printMatrix(A, nRows_A, nCols_A);
	printf("Matrix B\n");
	printMatrix(B, nRows_B, nCols_B);
  printf("Matrix C\n");
	printMatrix(C, nRows_A, nCols_B);

	float *dev_A, *dev_B, *dev_C;
	c_e = cudaMalloc((void **)&dev_A, nRows_A * nCols_A * sizeof(float));
	if(c_e!=cudaSuccess) {
		printf("Error (dev_A allocation): %d\n",c_e);
		exit(-1);
	}
	c_e = cudaMalloc((void **)&dev_B, nRows_A * nCols_A * sizeof(float));
	if(c_e!=cudaSuccess) {
		printf("Error (dev_B allocation): %d\n",c_e);
		exit(-1);
	}
	c_e = cudaMalloc((void **)&dev_C, nRows_A * nCols_B * sizeof(float));
	if(c_e!=cudaSuccess) {
		printf("Error (dev_C allocation): %d\n",c_e);
		exit(-1);
	}

	// Copy matrices to device
	cudaMemcpy(dev_A, A, nRows_A * nCols_A * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, nRows_B * nCols_B * sizeof(float), cudaMemcpyHostToDevice);

	// Execute kernel A
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);

	cudaEventRecord(start_kernel, 0);
	// cublas_matmul<<<Blocks,Threads>>>(dev_A, dev_B, dev_C, nRows_A, nCols_A, nCols_B);
  cublas_matmul (dev_A, dev_B, dev_C, nRows_A, nCols_A, nCols_B);
	cudaEventRecord(stop_kernel, 0);
	cudaEventSynchronize(stop_kernel);

	c_e=cudaThreadSynchronize();
	if(c_e!=cudaSuccess) {
		printf("Error: %d\n",c_e);
		exit(-1);
	}

	// Copy output matrix to host
	cudaMemcpy(C, dev_C, nRows_A * nCols_B * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Matrix C\n");
	printMatrix(C, nRows_A, nCols_B);

	// Destory events
	cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
	cudaEventDestroy(start_kernel);
	cudaEventDestroy(stop_kernel);

	// Release device memory
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	// Release host memory
	free(A);
	free(B);
	free(C);
}
