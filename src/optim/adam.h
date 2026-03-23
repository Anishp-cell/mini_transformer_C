#ifndef ADAM_H
#define ADAM_H

typedef struct{
    float *m;
    float *v;
    int size;
}AdamParam;

typedef struct{
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;
}AdamOptimizer;

void adam_init_param(AdamParam *p, int size);
void adam_free_param(AdamParam *p);

void adam_init(AdamOptimizer *opt, float lr);

void adam_update(AdamOptimizer *opt,AdamParam *p,float *weights,float *grads);

#endif