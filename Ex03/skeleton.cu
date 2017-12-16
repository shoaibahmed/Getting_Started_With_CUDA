#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define LLength 500
#define ITER 100
#define NUMADDS 1
#define NUMMEM 1
#define LNum 1

__global__ void kern_A(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	float a;
	int ia;
	for(ia=0;ia<ITER;ia++) 
	{
		a=A[i];
		for(j=0;j<LLength;j++) 
		{
			a=a+b;
		}
	}
	if(i==0) *B=a*1.0f;
}

__global__ void kern_M(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	float a;
	int ia;
	for(ia=0;ia<ITER;ia++) {
		a=A[i];
		for(j=0;j<LLength;j++) {
			a=A[(int)a];
		}}
	if(i==0) *B=a*1.0f;
}

__global__ void kern_C(int spacing,float *A,float *B,float b) {
	int i=threadIdx.x+blockIdx.x*spacing;
	int j;
	float a;
	int ia;
	for(ia=0;ia<ITER;ia++) {
		a=A[i];
		for(j=0;j<LLength;j++) {
			a=a+b;
			a=A[(int)a];
		}}
	if(i==0) *B=a*1.0f;
}

float control(float *A) {
	float a;
	a=A[0];
	int i,j;
	for(j=0;j<LLength;j++) {
		for(i=0;i<NUMMEM;i++) {
			a=A[(int)a];
		}
	}
	return(a);
}

int main()
{
	int Threads=32;
	int Warp=32;
	int Blocks=8;
	int Spacing=((LLength*LNum*NUMMEM+1)*Threads+Warp-1)/Warp * Warp;
	long BS,i,j,Bbegin;
	long size,OPS,MOPS;
	float k_time;
	cudaError_t c_e;
	cudaEvent_t start, stop;

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
	c_e=cudaMalloc((void **)&d_B,sizeof(float));
	if(c_e!=cudaSuccess) {
		printf("Fehler: %d\n",c_e);
		exit(-1);
	}
	for(i=0;i<=Blocks;i++) {
		Bbegin=Threads+Spacing*i;
		for(j=0;j<=Threads*LLength*LNum*NUMMEM;j++) {
			A[i*Spacing+j]=Bbegin+j;
		}
	} 
	c_e=cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	if(c_e!=cudaSuccess) {
		printf("Fehler: %d\n",c_e);
		exit(-1);
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	kern_A<<<Blocks,Threads>>>(Spacing,d_A,d_B,B);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	c_e=cudaThreadSynchronize();
	if(c_e!=cudaSuccess) {
		printf("Error : %d\n",c_e);
		exit(-1);
	}
	if(B!=control(A))
		printf("Error: Result is %.1f and should be  %.1f\n",B,control(A));

	cudaEventElapsedTime(&k_time, start, stop);
	printf("%ld ar. Operations and %ld Memory operations\n",OPS,MOPS);
	perf=(OPS*1.e-6)/(k_time*1.e-3);
	BW  =((long)(MOPS* sizeof(float))*1.e-06)/(k_time*1.e-3);
	printf("Performance: %.2f GFlops and %.2f GB/s\n",perf,BW);

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
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
