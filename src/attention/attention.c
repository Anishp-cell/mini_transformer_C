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
    attn->output= (float*)malloc(SEQ_LEN*D*sizeof(float));
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

void attention_forward(Attention *attn, float *input, float *output){
    int D= attn->embed_dim;
    for(int t=0; t<SEQ_LEN;t++){
        for(int d=0; d<D;d++){
            float q= 0.0f;
            float k= 0.0f;
            float v= 0.0f;
            for(int i=0;i<D;i++){
                float x= input[t*D +i]; // x is i-th dimension of t-th input token
                q+= x*attn->Wq[i*D+d];
                k+= x*attn->Wk[i*D+d];
                v+= x*attn->Wv[i*D+d];
            }
            attn->Q[t*D+d]=q;
            attn->K[t*D+d]=k;
            attn->V[t*D+d]=v;
        }
    }
    // compute attention scores = QK^T / sqrt(D)
    float scale= 1.0f/sqrt((float)D); // scale factor to prevent large dot products 
    for(int i=0;i<SEQ_LEN;i++){ //for each query token
        for(int j=0;j<SEQ_LEN;j++){ //for each key token of the same sequence
            float score= 0.0f; 
            for(int d=0;d<D;d++){ 
                score+= attn->Q[i*D+d] * attn->K[j*D+d]; 
            }
            attn->scores[i*SEQ_LEN+j]= score*scale;  // store unnormalized score
        }
    }
    // apply casual mask to prevent viewing future tokens
    apply_causal_mask(attn->scores); // this will set scores[i][j] to -inf for j>i, ensuring that the softmax will assign zero weight to future tokens
    // apply softmax to get attention weights (formula: weights[i][j]= exp(scores[i][j]) / sum_k exp(scores[i][k]))
    for(int i=0; i<SEQ_LEN;i++){
        float max_val=attn->scores[i*SEQ_LEN];
        for(int j=1;j<SEQ_LEN;j++){
            float val= attn->scores[i*SEQ_LEN+j];
            if(val>max_val) max_val=val;
        }
        float sum= 0.0f; 
        for(int j=0; j<SEQ_LEN;j++){
            float exp_val= expf(attn->scores[i*SEQ_LEN+j]-max_val); 
            attn->weights[i*SEQ_LEN+j]= exp_val; 
            sum+= exp_val;
        }
        for(int j=0;j<SEQ_LEN;j++){
           attn->weights[i * SEQ_LEN + j] /= (sum + 1e-9f); // normalize to get final attention weights
        }
    }
    //computing output as weighted sum of value vectors: output= weigths * V
    for(int i= 0;i<SEQ_LEN; i++){
        for(int d=0;d<D;d++){
            float val= 0.0f;
            for(int j=0;j<SEQ_LEN;j++){
                float w= attn->weights[i*SEQ_LEN+j]; // attention weight for j-th token
                val+= w * attn->V[j*D+d];
            }
            output[i*D+d]= val;
            attn->output[i*D+d]=val;
        }
    }
}
void attention_update(Attention *attn, float lr){
    int D= attn->embed_dim;
    for(int i=0;i<D*D;i++){
        attn->Wq[i]-= lr*attn->grad_Wq[i];
        attn->Wk[i]-= lr*attn->grad_Wk[i];
        attn->Wv[i]-= lr*attn->grad_Wv[i];
    }
}
void attention_free(Attention *attn)
{
    free(attn->Wq);
    free(attn->Wk);
    free(attn->Wv);

    free(attn->grad_Wq);
    free(attn->grad_Wk);
    free(attn->grad_Wv);

    free(attn->Q);
    free(attn->K);
    free(attn->V);

    free(attn->scores);
    free(attn->weights);

    free(attn->output);
}

