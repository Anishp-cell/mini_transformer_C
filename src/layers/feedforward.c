#define _USE_MATH_DEFINES
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "feedforward.h"
// gaussian random number generator using Box-Muller transform
static float rand_normal(){
    float u1 = (rand()+1.0f)/(RAND_MAX+1.0f);
    float u2 = (rand()+1.0f)/(RAND_MAX+1.0f);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);
}
void feedforward_init(FeedForward *ff){
    ff->embed_dim=EMBED_DIM;
    ff->ff_dim=FF_DIM; //4 times the embedding dimension
    int D= EMBED_DIM;
    int F= FF_DIM;
    //allocate weights and gradients
    ff->W1= (float*)malloc(D*F*sizeof(float));
    ff->b1=(float*)malloc(F*sizeof(float));
    ff->W2=(float*)malloc(D*F*sizeof(float));
    ff->b2=(float*)malloc(D*sizeof(float));
    ff->grad_W1=(float*)malloc(D*F*sizeof(float));
    ff->grad_b1= (float*)malloc(F*sizeof(float));
    ff->grad_W2=(float*)malloc(D*F*sizeof(float));
    ff->grad_b2=(float*)malloc(D*sizeof(float));
    ff->hidden=(float*)malloc(SEQ_LEN*F*sizeof(float));// hidden layer output before activation
    ff->hidden_act=(float*)malloc(SEQ_LEN*F*sizeof(float)); // hiddnen layer output after activation
    //initializing weights 
    for(int i=0;i<D*F;i++)
        ff->W1[i]=rand_normal()*INIT_STD; //init sdt = .2
    for(int i=0;i<F*D;i++)
        ff->W2[i]=rand_normal()*INIT_STD;
        memset(ff->b1,0,F*sizeof(float));
        memset(ff->b2,0,D*sizeof(float));
        printf("feedforward layer initiialized with embed_dim=%d, ff_dim=%d\n",ff->embed_dim,ff->ff_dim);
}
void feedforward_zero_grad(FeedForward *ff){
    int D= ff->embed_dim;
    int F= ff->ff_dim;
    memset(ff->grad_W1, 0,D*F*sizeof(float));
    memset(ff->grad_b1, 0,F*sizeof(float));
    memset(ff->grad_b2,0,D*sizeof(float));
    memset(ff->grad_W2,0, F*D*sizeof(float));
}
void feedforward_forward(FeedForward *ff, float *input, float *output){
    int D= ff->embed_dim;
    int F= ff->ff_dim;
    //input * W1 +b1 = hidden layer output before activation
    for(int i=0;i<SEQ_LEN;i++){
        for(int j=0;j<F;j++){
            float sum=ff->b1[j]; // start with bias
            for(int k=0;k<D;k++){
                sum+= input[i*D+k] * ff->W1[k*F+j];
            }
            ff->hidden[i*F+j]=sum;
            //ReLU activation
            ff->hidden_act[i*F+j]= (sum>0)? sum:0.0f;
        }
    }
    for(int t=0;t<SEQ_LEN;t++){
        for(int d=0;d<D;d++){
            float sum= ff->b2[d];
            for(int f=0;f<F;f++){
                sum+=ff->hidden_act[t*F+f]*ff->W2[f*D+d];
            }
            output[t*D+d]=sum;
        }
    }
}
void feedforward_backward(
    FeedForward *ff,
    float *input,
    float *d_output,
    float *d_input)
{
    int D = ff->embed_dim;
    int F = ff->ff_dim;

    float *d_hidden = calloc(SEQ_LEN * F, sizeof(float));

    /* backprop W2 */
    for(int t=0;t<SEQ_LEN;t++){
        for(int f=0;f<F;f++){
            for(int d=0;d<D;d++){
                float grad= d_output[t*D + d];
                ff->grad_W2[f*D + d]+= ff->hidden_act[t*F + f] * grad;
                ff->grad_b2[d]+= grad;
                d_hidden[t*F + f]+= grad * ff->W2[f*D + d];
            }
        }
    }

    // relu backward
    for(int i=0;i<SEQ_LEN*F;i++){
        if(ff->hidden[i] <= 0.0f)
            d_hidden[i] = 0.0f;
    }
    // backprop of W1 
    for(int t=0;t<SEQ_LEN;t++){
        for(int d=0;d<D;d++){
            for(int f=0;f<F;f++){
                float grad = d_hidden[t*F + f];
                ff->grad_W1[d*F + f]+=input[t*D + d] * grad;
                ff->grad_b1[f]+= grad;

                d_input[t*D + d]+= grad * ff->W1[d*F + f];
            }
        }
    }

    free(d_hidden);
}

void feedforward_update(FeedForward *ff, float lr){

    int D = ff->embed_dim;
    int F = ff->ff_dim;
    for(int i=0;i<D*F;i++)
        ff->W1[i] -= lr * ff->grad_W1[i];
    for(int i=0;i<F*D;i++)
        ff->W2[i] -= lr * ff->grad_W2[i];
    for(int i=0;i<F;i++)
        ff->b1[i] -= lr * ff->grad_b1[i];
    for(int i=0;i<D;i++)
        ff->b2[i] -= lr * ff->grad_b2[i];
}

void feedforward_free(FeedForward *ff){
    free(ff->W1); free(ff->b1);
    free(ff->W2); free(ff->b2);
    free(ff->grad_W1); free(ff->grad_b1);
    free(ff->grad_W2); free(ff->grad_b2);
    free(ff->hidden);
    free(ff->hidden_act);
}