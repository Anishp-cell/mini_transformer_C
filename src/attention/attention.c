#include<stdio.h>
#include<stdlib.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include<string.h>
#include "attention.h"
#include "mask.h"

// gaussian random number generator for weight initialization
static float rand_normal(){
    float u1= (rand()+1.0f)/(RAND_MAX+1.0f);
    float u2= (rand()+1.0f)/(RAND_MAX+1.0f);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);
}

// attention initialization
void attention_init(Attention *attn){
    int D= attn->embed_dim;
    // allocate projection weights and gradients
    attn->Wq= (float*)malloc(D*D*sizeof(float));
    attn->Wk= (float*)malloc(D*D*sizeof(float));
    attn->Wv= (float*)malloc(D*D*sizeof(float));

    attn->grad_Wq= (float*)malloc(D*D*sizeof(float));
    attn->grad_Wk= (float*)malloc(D*D*sizeof(float));
    attn->grad_Wv= (float*)malloc(D*D*sizeof(float));

    // allocate cached tensors
    attn->Q= (float*)malloc(SEQ_LEN*D*sizeof(float));
    attn->K= (float*)malloc(SEQ_LEN*D*sizeof(float));
    attn->V= (float*)malloc(SEQ_LEN*D*sizeof(float));

    attn->scores= (float*)malloc(SEQ_LEN * SEQ_LEN *sizeof(float));
    attn->weights= (float*)malloc(SEQ_LEN * SEQ_LEN * sizeof(float));
    attn->weights= (float*)malloc(SEQ_LEN*D*sizeof(float));
    // initialize weights
    for(int i=0;i<D*D;i++){
        attn->Wq[i]= rand_normal()*INIT_STD;
        attn->Wk[i]= rand_normal()*INIT_STD;
        attn->Wv[i]= rand_normal()*INIT_STD;
    }
    printf("attention initialized \n");

}

void attention_zero_grad(Attention *attn){
    int D= attn->embed_dim;
    memset(attn->grad_Wq, 0, D*D*sizeof(float));
    memset(attn->grad_Wk, 0, D*D*sizeof(float));
    memset(attn->grad_Wv,0, D*D*sizeof(float));
}
