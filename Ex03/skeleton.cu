#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define LLength 500
#define ITER 100
#define NUMADDS 31
#define NUMMEM 31
#define LNum 1

#define MAX_BLOCKS 8
#define MAX_THREADS 1024
#define THREAD_STRIDE 32

__global__ void kern_A(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
#if NUMADDS == 1
	volatile float a;
#elif NUMADDS == 2
	volatile float a, x0;
#elif NUMADDS == 3
	volatile float a, x0, x1;
#elif NUMADDS == 4
	volatile float a, x0, x1, x2;
#elif NUMADDS == 5
	volatile float a, x0, x1, x2, x3;
#else
	volatile float a, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;
#endif
#if NUMADDS >= 12
	volatile float x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
#endif
#if NUMADDS >= 22
	volatile float x20, x21, x22, x23, x24, x25, x26, x27, x28, x29;
#endif
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
#if NUMADDS >= 1
			a=a+b;
#endif
#if NUMADDS >= 2
			x0 += b;
#endif
#if NUMADDS >= 3
			x1 += b;
#endif
#if NUMADDS >= 4
			x2 += b;
#endif
#if NUMADDS >= 5
			x3 += b;
#endif
#if NUMADDS >= 6
			x4 += b;
			x5 += b;
			x6 += b;
			x7 += b;
			x8 += b;
			x9 += b;
#endif
#if NUMADDS >= 12
			x10 += b;
			x11 += b;
			x12 += b;
			x13 += b;
			x14 += b;
			x15 += b;
			x16 += b;
			x17 += b;
			x18 += b;
			x19 += b;
#endif
#if  NUMADDS >= 22
			x20 += b;
			x21 += b;
			x22 += b;
			x23 += b;
			x24 += b;
			x25 += b;
			x26 += b;
			x27 += b;
			x28 += b;
			x29 += b;
#endif
		}
	}
#if NUMADDS == 1
	if(i==0) *B=a*1.0f;
#elif NUMADDS == 2
	if(i==0) *B=a*x0*1.0f;
#elif NUMADDS == 3
	if(i==0) *B=a*x0*x1*1.0f;
#elif NUMADDS == 4
	if(i==0) *B=a*x0*x1*x2*1.0f;
#elif NUMADDS == 5
	if(i==0) *B=a*x0*x1*x2*x3*1.0f;
#elif NUMADDS == 11
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*1.0f;
#elif NUMADDS == 21
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*x10*x11*x12*x13*x14*x15*x16*x17*x18*x19;
#elif NUMADDS == 31
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*1.0f;
#endif
}

__global__ void kern_M(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
#if NUMADDS == 1
	volatile float a;
#elif NUMADDS == 2
	volatile float a, x0;
#elif NUMADDS == 3
	volatile float a, x0, x1;
#elif NUMADDS == 4
	volatile float a, x0, x1, x2;
#elif NUMADDS == 5
	volatile float a, x0, x1, x2, x3;
#else
	volatile float a, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;
#endif
#if NUMADDS >= 12
	volatile float x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
#endif
#if NUMADDS >= 22
	volatile float x20, x21, x22, x23, x24, x25, x26, x27, x28, x29;
#endif
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
#if NUMMEM >= 1
			a=A[(int)a];
#endif
#if NUMMEM >= 2
			x0=A[(int)x0];
#endif
#if NUMMEM >= 3
			x1=A[(int)x1];
#endif
#if NUMMEM >= 4
			x2=A[(int)x2];
#endif
#if NUMMEM >= 5
			x3=A[(int)x3];
#endif
#if NUMMEM >= 6
			x4=A[(int)x4];
			x5=A[(int)x5];
			x6=A[(int)x6];
			x7=A[(int)x7];
			x8=A[(int)x8];
			x9=A[(int)x9];
#endif
#if NUMMEM >= 12
			x10=A[(int)x10];
			x11=A[(int)x11];
			x12=A[(int)x12];
			x13=A[(int)x13];
			x14=A[(int)x14];
			x15=A[(int)x15];
			x16=A[(int)x16];
			x17=A[(int)x17];
			x18=A[(int)x18];
			x19=A[(int)x19];
#endif
#if NUMMEM >= 22
			x20=A[(int)x20];
			x21=A[(int)x21];
			x22=A[(int)x22];
			x23=A[(int)x23];
			x24=A[(int)x24];
			x25=A[(int)x25];
			x26=A[(int)x26];
			x27=A[(int)x27];
			x28=A[(int)x28];
			x29=A[(int)x29];
#endif
		}
	}
#if NUMMEM == 1
	if(i==0) *B=a*1.0f;
#elif NUMMEM == 2
	if(i==0) *B=a*x0*1.0f;
#elif NUMMEM == 3
	if(i==0) *B=a*x0*x1*1.0f;
#elif NUMMEM == 4
	if(i==0) *B=a*x0*x1*x2*1.0f;
#elif NUMMEM == 5
	if(i==0) *B=a*x0*x1*x2*x3*1.0f;
#elif NUMMEM == 11
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*1.0f;
#elif NUMMEM == 21
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*x10*x11*x12*x13*x14*x15*x16*x17*x18*x19;
#elif NUMMEM == 31
	if(i==0) *B=a*x0*x1*x2*x3*x4*x5*x6*x7*x8*x9*x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*1.0f;
#endif
}

__global__ void kern_C(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	volatile float a;
	int ia;
	for(ia=0; ia<ITER; ia++) {
		a=A[i];
		for(j=0; j<LLength; j++) {
			// Fixed LNum = 4
			a=a+b;
			a=a+b;
			a=a+b;
			a=a+b;
			
			a=A[(int)a];
			a=A[(int)a];
			a=A[(int)a];
			a=A[(int)a];
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
	int BlockConfigs[2] = {8, 8};
	int ThreadConfigs[2] = {128, 256};
	printf ("Executing kernels for Task # 02 and Task # 03\n");

	for (int configIter = 0; configIter < 2; configIter++)
	{
		// int Blocks = BlockConfigs[configIter] * props.multiProcessorCount;
		// int Threads = ThreadConfigs[configIter] * props.multiProcessorCount;
		int Blocks = BlockConfigs[configIter];
		int Threads = ThreadConfigs[configIter];

		int Spacing=((LLength*LNum*NUMMEM+1)*Threads+Warp-1)/Warp * Warp;
		long BS,i,j,Bbegin;
		long size,OPS,MOPS;
		float k_A_time, k_M_time;
		cudaError_t c_e;
		cudaEvent_t start_kernel_A, stop_kernel_A;
		cudaEvent_t start_kernel_M, stop_kernel_M;

		float *A,*d_A,*d_B;
		float B=0.0f;
		float perf, BW;
		BS=1024*1024*256;
		size=sizeof(float)*BS;
		i=(long)(Blocks*Spacing)+Threads*LLength*LNum*NUMMEM;
		OPS=(long)((long)(Blocks*Threads*LLength)/1024*LNum*NUMADDS*ITER);
		MOPS=(long)((long)(Blocks*Threads*LLength)/1024*LNum*NUMMEM*ITER);
		//printf("%ld = %d*%d*%d*%d*%d*%d\n",MOPS,Blocks,Threads,LLength,LNum,NUMMEM,ITER);
		if(i>BS) {
			printf(" To large: \n %ld (Index) vs. %ld (Allocated)\n",i,BS);
			exit(-1);
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
			for(j=0; j<=Threads*LLength*LNum*NUMMEM; j++) {
				A[i*Spacing+j]=Bbegin+j;
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

		// Copy back the results
		c_e=cudaMemcpy(&B,d_B,sizeof(float),cudaMemcpyDeviceToHost);
		if(B!=control(A))
			printf("Error: Result is %.1f and should be  %.1f (no error for kern_A)\n",B,control(A));

		cudaEventElapsedTime(&k_M_time, start_kernel_M, stop_kernel_M);

		printf("%ld ar. Operations and %ld Memory operations\n", OPS, MOPS);
		BW  =((long)(MOPS* sizeof(float))*1.e-06)/(k_M_time*1.e-3);
		perf=(OPS*1.e-6)/(k_A_time*1.e-3);
		printf("Performance: %.2f GFlops and %.2f GB/s\n",perf,BW);

		if (configIter == 0)
			fprintf (outputFileSecondTask, "%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);
		else
			fprintf (outputFileThirdTask, "%d,%d,%ld,%ld,%f,%f,%f,%f\n", NUMADDS, NUMMEM, OPS, MOPS, k_A_time, k_M_time, perf, BW);
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
		cudaEventDestroy(stop_kernel_A);
		cudaEventDestroy(stop_kernel_M);
	}

	fclose(outputFileFirstTask);
	fclose(outputFileSecondTask);
	fclose(outputFileThirdTask);
}
