#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "net_include.h"
#include "protos.h"

struct net_struct network;
struct layers *net;
int net_init_c=0;

void net_init(int layer, int *n_in)
{
 int i,j,len;
 float *aux;
 float fact;
 if(! net_init_c) {
  net_init_c=1;
  network.layer=layer;
  s_alloc(layer,&network.net);
  i_alloc(layer,&network.lc);
  network.lc[0]=0;
  net=network.net;
  for(i=0;i<layer;i++) net[i].n=n_in[i];  // copy input (neurons per layer)
  for(i=1;i<layer;i++) network.lc[i]=network.lc[i-1]+n_in[i]; // sum all but input
  network.l_sum=network.lc[layer-1];
  len=network.l_sum;
  // initialize bias and der_b on all but input layer
  f_alloc(2*network.l_sum,&aux);
  Fgaussrandvec(len,aux);
  net[0].b=NULL; // set bias of input to NULL pointer
  net[0].der_b=NULL;
  // set pointer per layer
  for(i=1,j=0;i<layer;i++) {
      net[i].b=&aux[j];
      net[i].der_b=&aux[len+j];
      j+=n_in[i];
  }
  // allocate space for activation (biases + input)
  len+=net[0].n;
  f_alloc(len,&aux);
  for(i=0,j=0;i<layer;i++) {
   net[i].activation=&aux[j];
   j+=net[i].n;
  }
  // initialize weight and der_w starting at 1
  for(i=0,j=0;i<layer-1;i++) j+=n_in[i]*n_in[i+1];
  len=j;
  network.l_prod=j;
  f_alloc(2*j,&aux); // reuse aux - pointer is saved above
  Fgaussrandvec(j,aux);
  net[0].w_in=NULL; // set weight to input to NULL pointer
  net[0].der_w=NULL;
  /* Organization w_in and der_w: 
   *              row all weights for a neuron 
   *              column different neurons
   * ex: layer 0 has 2 neurons, layer 1 has 3, then
   *     w_in(3,2), leading dimension and length in row is net[i-1].n (=2)
   *                # of rows is net[i].n (=3)
   */
  for(i=0,j=0;i<layer-1;i++) {
      net[i+1].w_in=&aux[j];
      net[i+1].der_w=&aux[len+j];
      j+=n_in[i]*n_in[i+1];
  }
  // Modification: division by sqrt(layer)
  for(i=0;i<layer-1;i++) {
   fact=1.e0/sqrt(n_in[i]*1.e0);
   for(len=0;len<n_in[i];i++) {
    for(j=0;j<n_in[i+1];j++) {
     net[i+1].w_in[len*n_in[i+1]+j] *= fact;
   }}}
   
 
  // largest layer
  for(i=0,j=0;i<layer;i++) {if(n_in[i]>j) j=n_in[i];}
  network.large=j;
  f_alloc(j,&network.a_l);
  f_alloc(network.l_sum*2,&network.a_sum2);
 } // end net_init_c
}

void compare(int *corr_f,int num,float *in, float *out)
{
 int layer=network.layer;
 int i,j,k,n,m,kv,corr=0;
 float v;
 n=net[0].n;
 m=net[layer-1].n;
 for(i=0;i<num;i++) {
  v=0.;
  kv=-1;
  feedfor(&in[n*i]);
  for(k=-1,j=0;j<m;j++) { 
      if(out[m*i+j]>0.1) k=j;
      if(net[layer-1].activation[j]>v) {v=net[layer-1].activation[j];kv=j;}
      // printf("%.3f ",net[layer-1].activation[j]);
  }     
  //printf(" gibt %d zu %d\n",kv,k);
  if(kv==k) corr++;
 }
 printf("%d correct out of %d\n",corr,num);
 *corr_f=corr;
}

void feedfor(float *in) {
 // apply <- sigma(<weight,act>+bias)
 int i,j,k;
 float *s,*b,*w,*act,*a;
 int layer=network.layer;
 s=network.a_l;
 a=net[0].activation;
 for(i=0;i<net[0].n;i++) a[i]=in[i]; // copy input
 for(i=1;i<layer;i++) {
  act=net[i].activation;
  b=net[i].b;
  w=net[i].w_in;
  a=net[i-1].activation;
  for(j=0;j<net[i].n;j++) {
   s[j]=b[j];
   for(k=0;k<net[i-1].n;k++) {
    s[j]+=w[j*net[i-1].n+k]*a[k];
   }
  }
  sigmoid(net[i].n,act,s);
 }
}

void grad(float *input,float *output) {
 // Get result for input and store intermediate values
 float *d_b,*zval,*zc,*z,s;
 float *activation,*b,*w,*der_b,*der_w,der;
 int i,j,k,iz;
 int nm,nn,np;
 int layer=network.layer;
 int lb=network.l_sum;

 d_b=&network.a_sum2[0];
 zval=&d_b[lb];
 zc=network.a_l;

 // copy input
 activation=net[0].activation;
 for(i=0;i<net[0].n;i++) {
  activation[i]=input[i];
 }
 for(i=1;i<layer;i++) {
  iz=network.lc[i-1];
  z=&zval[iz];
  activation=net[i-1].activation;
  w=net[i].w_in;
  b=net[i].b;
  nn=net[i].n;
  nm=net[i-1].n;
  // Matrix W times vector activation + vector b
  for(j=0;j<nn;j++) {
   s=b[j];
   for(k=0;k<nm;k++) {
    s+=w[j*nm+k]*activation[k];
   }
   z[j]=s;
  }
  // calculate activation for layer i
  activation=net[i].activation;
  // elementwise sigmafunction of vector zval
  sigmoid(nn,activation,z);
 }

 /* backward  last layer */
 j=layer-1;
 iz=network.lc[j-1];
 nn=net[j].n;
 z=&zval[iz];
 der_b=net[j].der_b;
 der_w=net[j].der_w;
 activation=net[j].activation;
 sigmoid_p(nn,zc,z); // elementwise sigma(prima) in vector zc
 for(i=0;i<nn;i++) {
  d_b[iz+i] =(activation[i]-output[i])*zc[i]; // elementwise (!)
  der_b[i]+=d_b[iz+i];  // sum over all in batch
//  printf("Abl %d: db %.2f a %.2f o %.2f zc %.2f zv %.2f\n",i,der_b[i],activation[i],output[i],zc[i],zval[iz+i]);
 }
 // derivative of w
 nm=net[j-1].n;
 activation=net[j-1].activation;
 for(j=0;j<nn;j++) {
 for(i=0;i<nm;i++) {
   der_w[j*nm+i]+=d_b[iz+j]*activation[i]; //rank-1 update
  }
 }

  //printf("\n");
 for(i=layer-2;i>0;i--) {
  nn=net[i].n;
  np=net[i+1].n;
  nm=net[i-1].n;
  iz=network.lc[i-1];
  z=&zval[iz];
  der_b=net[i].der_b;
  w=net[i+1].w_in;
  der_w=net[i].der_w;
  activation=net[i-1].activation;

  sigmoid_p(nn,zc,z); // vec(:nn)
  // w(T) * grad(b (layer+1))
  for(j=0;j<nn;j++) {
   for(k=0,s=0.e0;k<np;k++) {
    s+=w[k*nn+j]*d_b[network.lc[i]+k];
   }
   d_b[iz+j]=s*zc[j]; // new gradient (current layer)
   der_b[j]+=d_b[iz+j];
  }

  //rank -1 update
  for(k=0;k<nn;k++) {
  for(j=0;j<nm;j++) {
    der_w[k*nm+j]+=d_b[iz+k]*activation[j];
   }
  }
 }
}

void batch(int mb, float *in, float *res)
{
 int i,ji=0,jo=0,inl,ol,lb=0,lw=0;
 float *der_b,*der_w,*b,*w;
 int layer=network.layer;
 
 lb=network.l_sum;
 lw=network.l_prod;
 // set gradients to 0. Both are allocated linear
 for(i=0;i<lb;i++) net[1].der_b[i]=0.e0;
 for(i=0;i<lw;i++) net[1].der_w[i]=0.e0;
 inl=net[0].n;
 ol=net[layer-1].n;
 // calculate gradients and add them to values in net.der_b, net.der_w
 for(i=0;i<mb;i++) {
  grad(&in[ji],&res[jo]);
  ji+=inl;
  jo+=ol;
 }
 // calculate new biases and weights
 b=net[1].b;
 der_b=net[1].der_b;
 w=net[1].w_in;
 der_w=net[1].der_w;
 for(i=0;i<lb;i++) {
  b[i]=b[i]-network.eta/(mb*1.e0)*der_b[i];
 }
 for(i=0;i<lw;i++) {
  w[i]=w[i]-network.eta/(mb*1.e0)*der_w[i];
 }
}

void net_clear()
{
 if(net_init_c) {
  free(net[1].w_in);
  free(net[1].b);
  free(net[0].activation);
  free(network.net);
//printf("Fertg mit net jetzt a_l\n");
  free(network.a_l);
//printf("Jetzt a_sum2\n");
  free(network.a_sum2);
//printf("Jetzt lc\n");
  free(network.lc);
//printf("Fertg in net_clear\n");
  net_init_c=0;
 }
}
