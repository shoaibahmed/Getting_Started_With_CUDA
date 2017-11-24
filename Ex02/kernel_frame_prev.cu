#include <stdio.h>

__global__ void Yourkernel(uchar3 *dary,float t,int DIMX,int DIMY)
{
	/* Insert your kernel here */
  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int offset = blockId * blockDim.x + threadId;
  uchar3 color;
  if ((threadId < (DIMX / 2)) && (blockId < (DIMY / 2)))
    color = make_uchar3(255,0,0);
  else if ((threadId > (DIMX / 2)) && (blockId < (DIMY / 2)))
    color = make_uchar3(0,255,0);
  else if ((threadId < (DIMX / 2)) && (blockId > (DIMY / 2)))
    color = make_uchar3(255,255,0);
  else
    color = make_uchar3(0,0,255);
  dary[offset] = color;
}

void simulate(uchar3 *ptr, int tick, int w, int h)
{
	/* ptr is a pointer to an array of size w*h*sizeof(uchar3).
	   uchar3 is a structure with x,y,z coordinates to contain
	   red,yellow,blue - values for a pixel (Range [0,255])
	*/
	cudaError_t err=cudaSuccess;
	cudaEvent_t start,stop;
	float elapsedtime;

	cudaEventCreate  ( &start);
	cudaEventCreate  ( &stop);

	cudaEventRecord(start);
	/* Space for
	Yourkernel
	*/
  Yourkernel<<<h,w>>>(ptr, tick, w, h);
	err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n",
				 cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime, start, stop);
	printf("Time used: %.1f (ms)\n",elapsedtime);

	cudaEventDestroy  ( start);
	cudaEventDestroy  ( stop);

	printf("Please type ESC in graphics and afterwards RETURN in cmd-screen to finish\n");
}
