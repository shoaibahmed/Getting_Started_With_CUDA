#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define LLength 500
#define ITER 100
#define NUMADDS 32
#define NUMMEM 32
#define LNum 4

#define MAX_BLOCKS 8
#define MAX_THREADS 1024
#define THREAD_STRIDE 32
#define SYSTEM_CONFIGS 3

__global__ void kern_A(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	int k;
	volatile float a;
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
			for (k=0; k<NUMADDS; k++)
				a=a+b;
		}
	}
	if(i==0) *B=a*1.0f;
}

__global__ void kern_M(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	int k;
	volatile float a;
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
			for (k=0; k<NUMMEM; k++)
				a=A[(int)a];
		}
	}
	if(i==0) *B=a*1.0f;
}

__global__ void kern_C(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	int k;
	int l;
	volatile float a;
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
			// Fixed LNum = 4
			for (k=0; k<LNum; k++)
			{
				for (l=0; l<NUMADDS; l++)
					a=a+b;
				a=A[(int)a];
			}
		}
	}
	if(i==0) *B=a*1.0f;
}

float control(float *A) {
	float a;
	a=A[0];
	int i,j;
	for(j=0; j<LLength; j++) {
		for(i=0; i<NUMMEM; i++) {
			a=A[(int)a];
		}
	}
	return(a);
}

int main()
{
	int device = 0;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	int Warp = props.warpSize;
	int maxPossibleWarps = props.maxThreadsPerMultiProcessor / Warp;
	printf ("Warp size: %d | Steaming SMs: %d | Maximum threads per SM: %d | Max possible warps per SM: %d\n", Warp, props.multiProcessorCount, props.maxThreadsPerMultiProcessor, maxPossibleWarps);

	FILE* outputFileFirstTask = fopen("Output-task01.txt", "w");
	FILE* outputFileSecondTask = fopen("Output-task02.txt", "a");
	FILE* outputFileThirdTask = fopen("Output-task03.txt", "a");
	FILE* outputFileFourthTask = fopen("Output-task04.txt", "a");

	for (int block=1; block <= MAX_BLOCKS; block++)
	{
		for (int thread=THREAD_STRIDE; thread <= MAX_THREADS; thread += THREAD_STRIDE)
		{
			int Threads=thread;
			// int Blocks=block * props.multiProcessorCount;

			// Calculate theoretical occupancy
			int maxActiveBlocksKernelA, maxActiveBlocksKernelM;
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksKernelA, kern_A, Threads, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksKernelM, kern_C, Threads, 0);

			// Occupancy = active warps/Maximum active warps
			int currentPossibleActiveBlocksKernelA = min(block, maxActiveBlocksKernelA);
			int currentPossibleActiveBlocksKernelM = min(block, maxActiveBlocksKernelM);
			int currentPossibleActiveWarpsKernelA = (currentPossibleActiveBlocksKernelA * Threads) / Warp;
			int currentPossibleActiveWarpsKernelM = (currentPossibleActiveBlocksKernelM * Threads) / Warp;

			float occupancyKernelA = (float)currentPossibleActiveWarpsKernelA / maxPossibleWarps;
			float occupancyKernelM = (float)currentPossibleActiveWarpsKernelM / maxPossibleWarps;
			printf ("Blocks: %d | Threads: %d | Max active blocks (Kernel A): %d | Max active blocks (Kernel M): %d | Occupancy (Kernel A): %f | Occupancy (Kernel M): %f\n",
				block, Threads, maxActiveBlocksKernelA, maxActiveBlocksKernelM, occupancyKernelA, occupancyKernelM);
			fprintf (outputFileFirstTask, "%d,%d,%f,%f\n", block, Threads, occupancyKernelA, occupancyKernelM);
		}
	}

	// Parameters for 50% and 100% occupancy
	int BlockConfigs[SYSTEM_CONFIGS] = {8, 8, 8};
	int ThreadConfigs[SYSTEM_CONFIGS] = {128, 256, 128};
	printf ("Executing kernels for Task # 02, Task # 03 and Task # 04\n");

	for (int configIter = 0; configIter < SYSTEM_CONFIGS; configIter++)
	{
		printf ("Executing configuration %d\n", configIter+1);
		int Blocks = BlockConfigs[configIter] * props.multiProcessorCount;
		int Threads = ThreadConfigs[configIter];

		// int Spacing=((LLength*lNumMultiplier*NUMMEM+1)*Threads+Warp-1)/Warp * Warp;
		int Spacing = 0;
		long BS,i,j,Bbegin;
		long size,OPS,MOPS;
		float k_A_time, k_M_time, k_C_time;
		cudaError_t c_e;
		cudaEvent_t start_kernel_A, stop_kernel_A;
		cudaEvent_t start_kernel_M, stop_kernel_M;
		cudaEvent_t start_kernel_C, stop_kernel_C;

		float *A,*d_A,*d_B;
		float B=0.0f;
		float perf, BW;
		BS=1024*1024*256;
		size=sizeof(float)*BS;

		if (configIter == SYSTEM_CONFIGS - 1)
		{
			Spacing=((LLength*LNum+1)*Threads+Warp-1)/Warp * Warp;

			i=(long)(Blocks*Spacing)+Threads*LLength*LNum*1;
			OPS=(long)((long)(Blocks*Threads*LLength)/1024*LNum*NUMADDS*ITER);
			MOPS=(long)((long)(Blocks*Threads*LLength)/1024*LNum*ITER);
		}
		else
		{
			Spacing=((LLength*NUMMEM+1)*Threads+Warp-1)/Warp * Warp;

			i=(long)(Blocks*Spacing)+Threads*LLength*NUMMEM;
			OPS=(long)((long)(Blocks*Threads*LLength)/1024*NUMADDS*ITER);
			MOPS=(long)((long)(Blocks*Threads*LLength)/1024*NUMMEM*ITER);
		}
		
		//printf("%ld = %d*%d*%d*%d*%d*%d\n",MOPS,Blocks,Threads,LLength,LNum,NUMMEM,ITER);
		if(i>BS) {
			printf(" To large: \n %ld (Index) vs. %ld (Allocated)\n",i,BS);
			// exit(-1);
			printf(" Error occured. Skipping current step.\n");
			continue;
		} else {
			printf(" Sizes:  %ld (Index) vs. %ld (Allocated)\n",i,BS);
		}
		A=(float*)malloc(size);
		// printf("Allocated %ld Bytes\n",size);
		cudaSetDevice(0);
		c_e=cudaMalloc((void **)&d_A,size);
		if(c_e!=cudaSuccess) {
			printf("Error (d_A allocation): %d\n",c_e);
			exit(-1);
		}
		c_e=cudaMalloc((void **)&d_B,sizeof(float));
		if(c_e!=cudaSuccess) {
			printf("Error (d_B allocation): %d\n",c_e);
			exit(-1);
		}
		for(i=0; i<=Blocks; i++) {
			Bbegin=Threads+Spacing*i;
			if (configIter == SYSTEM_CONFIGS - 1) {
				for(j=0; j<=Threads*LLength*LNum*NUMMEM; j++) {
					A[i*Spacing+j]=Bbegin+j;
				}
			}
			else {
				for(j=0; j<=Threads*LLength*NUMMEM; j++) {
					A[i*Spacing+j]=Bbegin+j;
				}
			}
		}
		c_e=cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
		if(c_e!=cudaSuccess) {
			printf("Error (Copying A to d_A): %d\n",c_e);
			exit(-1);
		}
		
		// Execute kernel A
		cudaEventCreate(&start_kernel_A);
		cudaEventCreate(&stop_kernel_A);

		cudaEventRecord(start_kernel_A, 0);
		kern_A<<<Blocks,Threads>>>(Spacing,d_A,d_B,B);
		cudaEventRecord(stop_kernel_A, 0);
		cudaEventSynchronize(stop_kernel_A);

		c_e=cudaThreadSynchronize();
		if(c_e!=cudaSuccess) {
			printf("Error: %d\n",c_e);
			exit(-1);
		}

		cudaEventElapsedTime(&k_A_time, start_kernel_A, stop_kernel_A);
		
		// Execute kernel M
		cudaEventCreate(&start_kernel_M);
		cudaEventCreate(&stop_kernel_M);

		cudaEventRecord(start_kernel_M, 0);
		kern_M<<<Blocks,Threads>>>(Spacing,d_A,d_B,B);
		cudaEventRecord(stop_kernel_M, 0);
		cudaEventSynchronize(stop_kernel_M);

		c_e=cudaThreadSynchronize();
		if(c_e!=cudaSuccess) {
			printf("Error: %d\n",c_e);
			exit(-1);
		}

		// Execute kernel C
		cudaEventCreate(&start_kernel_C);
		cudaEventCreate(&stop_kernel_C);

		cudaEventRecord(start_kernel_C, 0);
		kern_C<<<Blocks,Threads>>>(Spacing,d_A,d_B,B);
		cudaEventRecord(stop_kernel_C, 0);
		cudaEventSynchronize(stop_kernel_C);

		c_e=cudaThreadSynchronize();
		if(c_e!=cudaSuccess) {
			printf("Error: %d\n",c_e);
			exit(-1);
		}

		cudaEventElapsedTime(&k_C_time, start_kernel_C, stop_kernel_C);

		// Copy back the results
		c_e=cudaMemcpy(&B,d_B,sizeof(float),cudaMemcpyDeviceToHost);
		if(B!=control(A))
			printf("Error: Result is %.1f and should be  %.1f (no error for kern_A)\n",B,control(A));

		cudaEventElapsedTime(&k_M_time, start_kernel_M, stop_kernel_M);

		if (configIter < SYSTEM_CONFIGS - 1)
		{
			BW=((long)(MOPS* sizeof(float))*1.e-06)/(k_M_time*1.e-3);
			perf=(OPS*1.e-6)/(k_A_time*1.e-3);
		}
		else
		{
			BW=((long)(MOPS* sizeof(float))*1.e-06)/(k_C_time*1.e-3);
			perf=(OPS*1.e-6)/(k_C_time*1.e-3);
		}

		printf("%ld ar. Operations and %ld Memory operations\n", OPS, MOPS);
		printf("Performance: %.2f GFlops and %.2f GB/s\n",perf,BW);

		if (configIter == 0)
			fprintf (outputFileSecondTask, "%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);
		else if (configIter == 1)
			fprintf (outputFileThirdTask, "%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);
		else
			fprintf (outputFileFourthTask, "%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);
		printf ("%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);

		/*
		printf("Array:\n");
		for(i=0;i<Blocks;i++) {
		printf("Block: %ld\n",i);
		for(j=0;j<Threads*LLength*LNum;j++) {
		printf("%.0f ",A[i*Spacing+j]);
		if((j+1)%10==0) printf("\n");
		}
		}
		*/

		cudaEventDestroy(start_kernel_A);
		cudaEventDestroy(start_kernel_M);
		cudaEventDestroy(start_kernel_C);
		cudaEventDestroy(stop_kernel_A);
		cudaEventDestroy(stop_kernel_M);
		cudaEventDestroy(stop_kernel_C);
	}

	fclose(outputFileFirstTask);
	fclose(outputFileSecondTask);
	fclose(outputFileThirdTask);
	fclose(outputFileFourthTask);
}
