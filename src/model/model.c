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
// zero out gradients before backprop
    void model_zero_grad(Model *m){
        int V= m->vocab_size;;
        int D= m->embed_dim;
        memset(m->embedding.grad_weight,0,V*D*sizeof(float));
        memset(m->output.grad_W,0,D*V*sizeof(float));
        memset(m->output.grad_b,0,V*sizeof(float));
    }
//forward pass
void model_forward(Model *m, uint16_t *input_tokens){
    int V= m->vocab_size;
    int D= m->embed_dim;
    for(int t=0;t<SEQ_LEN;t++){
        uint16_t token=input_tokens[t];
        for(int d=0;d<D;d++){
            m->embed_buffer[t*D+d]= m->embedding.weight[token*D+d];
        }
        for(int v=0;v<V;v++){
            float sum= m->output.b[v];
            for(int d= 0;d<D;d++){
                sum+=m->embed_buffer[t*D+d]*m->output.W[d*V+v];;
            }
            m->logit_buffer[t*V+v]=sum;
        }
    }
}
