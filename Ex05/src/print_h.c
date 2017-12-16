#include <stdio.h>
#include "net_include.h"
#include "protos.h"

void print_bw(float *b,float *w) {
 int i,j,k,ij;
 extern struct net_struct network;
 struct layers *net=network.net;
 int layer=network.layer;

 for(ij=0,i=1;i<layer;i++) {
  printf("b for layer %d\n",i);
  for(j=0;j<net[i].n;j++,ij++) {
   printf("%f ",b[ij]);
  }
  printf("\n");
 }
 for(ij=0,i=1;i<layer;i++) {
  printf("w between layer %d to %d\n",i-1,i);
  for(j=0;j<net[i].n;j++) {
   printf("(");
   for(k=0;k<net[i-1].n;k++,ij++) {
    printf("%f ",w[ij]);
   }
  printf(")\n");
  }
 }
}
