#include "net_include.h"
void net_init(int,int*);
void net_clear();
void feedfor(float*);
void grad(float*,float*);
void batch(int,float*,float*);
void train_it(int, int, int, struct sets, struct sets);

// helper
void i_alloc(int,int**);
void f_alloc(int,float**);
void s_alloc(int,struct layers**);
void Fgaussrandvec(int,float*);
void intrandvec(int,int*,int, int);
void sigmoid(int,float*,float*);
void sigmoid_p(int,float*,float*);
void compare(int*,int,float*,float*);
//
void print_bw();
void read_file(int*,char*,float**,float**);
int arguments(int , char**,int*, char**, char**, char**);
void s_batch(int,int,float*,int,float*,int,float*,float*);
void t_clear(struct sets);
void read_network(int*,int **,char *);
