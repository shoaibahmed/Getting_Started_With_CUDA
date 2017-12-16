#include <stdlib.h>
#include "protos.h"

void s_batch(int bs,int set, float *in,int ldi,float *out,int ldo,float *bi,float *bo)
{
 int *vec;
 int i,j;
 i_alloc(bs,&vec);
 intrandvec(bs,vec,0,set-1); // generate random in [0,set-1]
 for(i=0;i<bs;i++) {
  for(j=0;j<ldi;j++) {
   bi[i*ldi+j]=in[vec[i]*ldi+j];
  }
  for(j=0;j<ldo;j++) {
   bo[i*ldo+j]=out[vec[i]*ldo+j];
  }
 }
 free(vec);
}
