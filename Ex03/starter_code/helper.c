#ifndef __HELPER__
#define __HELPER__
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef __NET_INCLUDE__
#include "net_include.h"
#endif
inline void i_alloc(int cnt, int **P ) {
 *P=malloc(sizeof(**P)*cnt);
 if(*P==NULL) {
    printf("No space for allocation\n");
    exit(-1);
 }
}
inline void f_alloc(int cnt, float **P ) {
 *P=malloc(sizeof(**P)*cnt);
 if(*P==NULL) {
    printf("No space for allocation\n");
    exit(-1);
 }
}
inline void s_alloc(int cnt, struct layers **P ) {
 *P=malloc(sizeof(**P)*cnt);
 if(*P==NULL) {
    printf("No space for allocation\n");
    exit(-1);
 }
}
/* random numbers
*/
double Dgaussrand()
{
 double U1 = (double)rand() / RAND_MAX;
 return 2.e0*U1-1.e0;
}
void Fgaussrandvec(int n,float *vec)
{
 int i;
 for (i=0;i<n;i++) vec[i]=(float)Dgaussrand();
}
void intrandvec(int n,int *vec,int a, int b) {
 int i,range=b-a;
 for (i=0;i<n;i++) {
  double U =(double)rand()/RAND_MAX * range;
  vec[i] = (int) U + a;
 }
}
inline void sigmoid(int n,float *o,float *z) {
 int i;
 for(i=0;i<n;i++) {
  o[i]=1.0e0/(1.0e0+n*exp(-z[i]));
 }
}
inline void sigmoid_p(int n,float *o,float *z) {
 int i;
 sigmoid(n,o,z);
 for(i=0;i<n;i++) {
  o[i]=o[i]*(1.e0-o[i]);
 }
}
#endif
