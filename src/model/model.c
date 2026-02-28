#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include "model.h"

// box muller transform for normal distribution sampling
static float ran_normal(){
    float u1= (rand()+1.0f)/(RAND_MAX+1.0f);
    float u2= (rand()+1.0f)/(RAND_MAX+1.0f);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);

}

//model initialization
void model_init(Model *m){
    m->vocab_size= VOCAB_SIZE;
    m->embed_dim=EMBED_DIM;
 
    int V= m->vocab_size;
    int D= m->embed_dim;
       //allocate embedding weights and gradients
    m->embedding.weight= (float*)malloc(V*D*sizeof(float));
    m->embedding.grad_weight= (float*)malloc(V*D*sizeof(float));
    //allocate output layer and gradients
    m->output.W= malloc(D*V*sizeof(float));
    m->output.b= malloc(V*sizeof(float));
    m->output.grad_W= malloc(D*V*sizeof(float));
    m->output.grad_b= malloc(V*sizeof(float));
    //initialize weights with small random values
    for(int i=0; i<V*D;i++){
        m->embedding.weight[i]= ran_normal()* INIT_STD;
    } 
    for(int i= 0;i<D*V;i++){
        m->output.W[i]= ran_normal()*INIT_STD;
    }
    for(int i=0;i<V;i++){
        m->output.b[i]=0.0f;
    }
    
}