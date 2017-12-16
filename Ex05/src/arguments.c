#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "protos.h"
#define  TESTFILE data/test.dat
#define  TRAINFILE data/train.dat
#define  NETDEF data/netdef.inp

int arguments(int argc, char **argv,int *first, char **testf, char **trfile, char **nfile)
{
 int c;
 int height=28;
 int width=28;
 float eta;
 extern struct net_struct network;
 network.eta=ETA;
  while ((c = getopt (argc, argv, "e:n:t:p:h:w:")) != -1)
    switch (c)
      {
      case 'e':
        eta = atof(optarg);
        network.eta=eta;
        break;
      case 'n':
        *nfile = optarg;
        break;
      case 't':
        *trfile = optarg;
        break;
      case 'p':
        *testf= optarg;
        break;
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width= atoi(optarg);
        break;
      case '?':
          fprintf (stderr, "-p testdata -t traindata [-n network] [-h height] [-w width]\n");
          if (isprint (optopt)) {
           fprintf (stderr, "Unknown option `-%c'.\n", optopt);
           return -1;
          }
         break;
      default:
        abort ();
      }
   *first=width*height;
   if(*testf==NULL || *trfile==NULL) {
      fprintf(stderr,"-t and -p with argument required\n");
      return -1;
   }
#if DEBUG
   printf("Called: -p %s -t %s -h %d -w %d\n",*testf,*trfile,height,width);
#endif
   return 0;
}
void read_network(int *net_dim,int **n,char *filename) {
 int i,j,*p;
 FILE *fp;
 fp = fopen(filename, "r");
 if (fp == NULL) {
     fprintf(stderr, "Could not open file %s for reading\n", filename);
     exit(-1);
 }
 fscanf(fp,"%d",net_dim);
 printf("neuron layers %d\n",*net_dim);
 i_alloc(*net_dim,n);
 p=*n;
 for(i=0;i<*net_dim;i++) {
  fscanf(fp,"%d",&j);
  p[i]=j;
  printf("%d ",p[i]);
 }
 printf("\n");
 fclose(fp);
}

