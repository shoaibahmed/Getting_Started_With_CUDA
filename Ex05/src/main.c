#include <stdio.h>
#include <stdlib.h>
#include "net_include.h"
#include "protos.h"

FILE *fp_train = NULL;
FILE *fp_test  = NULL;

int main(int argc, char **argv) {
 int i,j;
// Number of neurons in 1st (=input) and last layer
 int first,last=10;
// extern struct net_struct *net;
 char *test_file = NULL;
 char *train_file = NULL;
 char *netdef_file= NULL;
 struct sets train,tests;

 train.number=2;
 tests.number=2;
 int set;
 int batchsize = 10;
 float *input,*output;

 i=arguments(argc,argv,&first, &test_file, &train_file, &netdef_file);
 if(i<0) {printf("Wrong arguments\n"); return -1; }

 read_file(&set,train_file,&input,&output);
 train.in=input;
 for(j=0,i=0;i<first;i++) {if(input[i]>0.1) {j=i;break;}};
/*
 printf("Kontrolle IN[%d] %f\n",j,input[j]);
 printf("Kontrolle T.IN[%d] %f\n",j,train.in[j]);
*/
 train.out=output;
 train.number=set;
 train.first=first;
 train.last=last;
 printf("Training sets: %d\n",set);

 // read test data
 read_file(&set,test_file,&input,&output);
 tests.in=input;
 tests.out=output;
 tests.number=set;
 tests.first=first;
 tests.last=last;
 printf("Test sets: %d\n",set);

 if(netdef_file == NULL) {
  printf("Standard eingabe\n");
  int n[3]={first,120,last};
  net_init(3,n);
 } else {
  int *n;
  int net_dim;
  read_network(&net_dim,&n,netdef_file);
/*
  printf("Main: layer: %d mit ",net_dim);
  for(i=0;i<net_dim;i++) printf("%d ",n[i]);
  printf("\n");
*/
  net_init(net_dim,n);
 }
 //feedfor(&input[0]);
  
 train_it(j,batchsize,100,train,tests);

 net_clear();
 t_clear(train);
 t_clear(tests);
 return(0);
}
