#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "protos.h"

float *openImage(int*,int , int , char*, FILE**);
int readImage(int,FILE*,int,int,int,float*, float*);

void read_file(int *sets, char* infile,float **in, float **out)
{
 extern FILE *fp_train;
 FILE *fp=fp_train;
 // define width and height of this single image here 
 int pic_width = 28;
 int pic_height = 28;
 int i,j;
 float *p;

 *in = openImage(sets,pic_width, pic_height,infile,&fp);
 if(*in ==NULL) goto finalise;
 j=*sets*10;
 f_alloc(j,out);
 p=*out;
 for(i=0;i<j;i++) { p[i]=0.0000;}
 readImage(*sets,fp,pic_width,pic_height,1,*in,*out);

 finalise:
 if(fp!=NULL) fclose(fp);
}

float *openImage(int *sets, int width, int height, char* filename, FILE **fp)
{
        float *code = NULL;
        int i=0;
        // Open file for reading (ascii file)
        *fp = fopen(filename, "r");
        if (*fp == NULL) {
                fprintf(stderr, "Could not open file %s for reading\n", filename);
                code = NULL;
                goto finalise;
        }

        char c[512];
        sprintf(c,"wc -l %s",filename);
        FILE *pp=popen(c,"r");
        if(pp==NULL) goto finalise;
        fscanf(pp,"%d %s",&i,c);
        fclose(pp);
        *sets=i/2;
        printf("%d training sets\n",*sets);

        float *buffer = (float *) malloc(width * height * (*sets) * sizeof(float));
        if (buffer == NULL) {
                fprintf(stderr, "Could not create image buffer\n");
                code = NULL;
                goto finalise;
        }

        return buffer;

        finalise:
        if (*fp != NULL) fclose(*fp);
        if (pp != NULL) fclose(pp);

        return code;
}

int readImage(int sets,FILE *fp,int width, int height, int num, float *buffer,float *number)
{
       int n,i,j,ij=0;

       for(j=0;j<sets;j++) {
        for (i=0;i<width*height;i++) {
         if(fscanf(fp,"%f",&buffer[ij])<=0) {
            fprintf(stderr,"Error reading element %d\n",ij);
            goto finalise;
         }
         ij++;
        }
        if(fscanf(fp,"%d",&n)<=0) {
            fprintf(stderr,"Error reading number %d\n",j);
            goto finalise;
        }
        number[j*10+n]=1.0;
       }

       return 0;

       finalise:
        if (fp != NULL) fclose(fp);
        return -1;
}
