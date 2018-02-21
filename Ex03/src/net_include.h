#ifndef __NET_INCLUDE__
#define __NET_INCLUDE__
struct layers{
 int n; 
 float *b, *w_in;
 float *der_b, *der_w, *activation;
};
struct net_struct {
 int layer; // #layers in net
 struct layers *net;
 int large; // neurons in largest layer
 float *a_l; // array of that size (scratch)
 int l_sum, l_prod; // sum and product of neurons in all (but 1st) layers
 float *a_sum2; // array (scratch) 2 times sum
 int *lc; // continuous numbering of neurons (0, +net[0].n, +net[1].n, ...)
 float eta;
};
struct sets {
 int first,last,number;
 float *in,*out;
};

#define ETA 2.0
#define IDENTIFIED 3000

#endif
