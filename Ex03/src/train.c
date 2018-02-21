#include <stdio.h>
#include <stdlib.h>
#include "protos.h"
#include "net_include.h"


void train_it(int j,int bs,int epochs,struct sets train,struct sets t)
{
 float *bi,*bo,*ti,*to;
 int i,cset,corr;
 int first=train.first, last=train.last, set=train.number;
 int it,k;

#ifdef DEBUG
 printf("Kontrolle T.IN[%d] %f\n",j,train.in[j]);
 
 // generate/copy a random test set in ti,to
 cset=10;
 f_alloc(cset*first,&ti);
 f_alloc(cset*last,&to);
 s_batch(cset,t.number,t.in,t.first,t.out,t.last,ti,to);
 printf("compare\n");
#else
 ti=t.in;
 to=t.out;
 cset=t.number;
#endif
 compare(&corr,cset,ti,to);

#ifdef DEBUG
 printf("allocating %d to bi\n",bs*first);
 printf("allocating %d to bo\n",bs*last);
 printf("Kontrolle T.IN[%d] %f\n",j,train.in[j]);
#endif
 f_alloc(bs*first,&bi);
 f_alloc(bs*last,&bo);


 for(i=0;i<epochs;i++) {
  // generate/copy a random training set in bi,bo
  s_batch(bs,set,train.in,first,train.out,last,bi,bo);
/*  printf("Testsatz:");
  for(it=0;it<bs;it++) {
  for(k=-1,j=0;j<last;j++) {
      if(bo[last*it+j]>0.1) k=j;
  }
  printf(" %d",k);
  }
  printf("\n");
*/
  // run this set to update weights and biases
  batch(bs,bi,bo);
  printf("Epoche %d\n",i+1);
  compare(&corr,cset,ti,to);
  if(corr > IDENTIFIED) break;
  
 }
 free(bi);
 free(bo);
#ifdef DEBUG
 free(ti);
 free(to);
#endif
}
void t_clear(struct sets t) {
 free(t.in);
 free(t.out);
}
