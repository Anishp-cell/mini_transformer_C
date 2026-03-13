#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "layernorm.h"

void layernorm_init(LayerNorm *ln){
    ln->embed_dim= EMBED_DIM;
    int D= ln->embed_dim;
    ln->gamma= (float*)malloc(D*sizeof(float));
    ln->beta= (float*)malloc(D*sizeof(float));
    ln->grad_gamma= (float*)malloc(D*sizeof(float));
    ln->grad_beta=  (float*)malloc(D*sizeof(float));
    ln->mean= (float*)malloc(SEQ_LEN*sizeof(float));
    ln->var= (float*)malloc(SEQ_LEN*sizeof(float));
    ln->x_hat= (float*)malloc(SEQ_LEN*D*sizeof(float));
    for(int d=0; d<D;d++){
        ln->gamma[d]=0.1f;
        ln->beta[d]=0.0f;
    }
    printf("layerNorm initialized\n");
}
//zero gradients for gamma and beta
void layernorm_zero_grad(LayerNorm *ln){
    int D= ln->embed_dim;
    memset(ln->grad_gamma, 0, D*sizeof(float));
    memset(ln->grad_beta, 0 , D*sizeof(float));
}
//forward pass of layer normalization where 
// input: x of shape( seq_len, embed_dim)
// output: out of shape( seq_len, embed_dim)

// d is the embedding dimension and seq_len is the number of tokens in the sequence 
// for each timestep(token position) t in the sequence, we compute the mean and variance of the input across the embedding dimension, 
// then normalize the input and apply the learnable parameters gamma and beta to produce the output

void layernorm_forward(LayerNorm *ln, float *input, float *output){
    int D= ln->embed_dim;
    
    // STEP 1: For each timestep (token position) in the sequence
    for(int t= 0; t<SEQ_LEN;t++){
        // Get pointer to the current token's embedding vector (length D)
        float *x= &input[t*D];
        
        // STEP 2: Compute mean across the embedding dimension
        // μ_t = (1/D) * Σ x_{t,d}
        float mean= 0.0f;
        for(int d=0;d<D;d++){
            mean+=x[d];
        }
        mean/=D;
        ln->mean[t]=mean;  // Store mean for use in backward pass
        
        // STEP 3: Compute variance across the embedding dimension
        // σ_t^2 = (1/D) * Σ (x_{t,d} - μ_t)^2
        float var= 0.0f;
        for(int d=0;d<D;d++){
            float diff= x[d]-mean;
            var+= diff*diff;
        }
        var/=D;
        ln->var[t]=var;  // Store variance for use in backward pass
        
        // STEP 4: Compute inverse standard deviation with epsilon for numerical stability
        // This avoids division by zero when variance is very small
        float inv_std= 1.0f/sqrtf(var+1e-5f); // 1 / sqrt(σ_t^2 + ε)
        
        // STEP 5: Normalize and apply learnable scale/shift for each dimension
        for(int d= 0;d<D;d++){
            // Normalize: x̂_{t,d} = (x_{t,d} - μ_t) / sqrt(σ_t^2 + ε)
            float x_hat= (x[d]-mean)*inv_std;
            ln->x_hat[t*D +d]= x_hat;  // Store normalized value for use in backward pass
            
            // Apply learnable parameters (scale and shift):
            // y_{t,d} = γ_d * x̂_{t,d} + β_d
            output[t*D +d]= ln->gamma[d]*x_hat + ln->beta[d];
        }
    }
}
void layernorm_backward(LayerNorm *ln, float *input, float *output_grad, float *input_grad){
    int D= ln->embed_dim;
    for(int t=0;t<SEQ_LEN;t++){
        //get pointers to current token's input, output and gradients
        float *xhat= &ln->x_hat[t*D]; // normalized input from forward pass
        float *gout= &output_grad[t*D]; // gradient from next layer w.r.t. output of this layer
        float *gin= &input_grad[t*D]; // gradient w.r.t. input to this layer (to be computed)
        float mean= ln->mean[t]; 
        float var= ln->var[t]; //mean and variance from fwd pass
        float inv_std= 1.0f / sqrt (var+1e-5f); // inverse std from fwd pass Epsilon(EPS) is 1e-5f
        for(int d=0;d<D;d++){
            ln->grad_gamma[d]+= gout[d]*xhat[d];
            ln->grad_beta[d]+= gout[d];
        }
        // compute g= dL/dy*gamma here g is gradient of the loss wrt normalized input xhat
        float sum_g=0.0f;
        float sum_g_xhat=0.0f;
        for(int d=0;d<D;d++){
            float g= gout[d]*ln->gamma[d];
            sum_g+=g;
            sum_g_xhat+= g*xhat[d];
        }
        //compute final gradient
        for(int d=0;d<D;d++){
            float g= gout[d]*ln->gamma[d];
            gin[d]=inv_std*(g-sum_g/D - xhat[d]*sum_g_xhat/D); // dL/dx= (1/sqrt(var+EPS))*(g - mean(g) - xhat*mean(g*xhat))
        }
    }
}
void layernorm_update(LayerNorm *ln, float lr){
    int D = ln->embed_dim;

    for(int d=0; d<D; d++){
        ln->gamma[d] -= lr * ln->grad_gamma[d];
        ln->beta[d]  -= lr * ln->grad_beta[d];
    }
}
void layernorm_free(LayerNorm *ln) {
    free(ln->gamma);
    free(ln->beta);
    free(ln->grad_gamma);
    free(ln->grad_beta);
    free(ln->mean);
    free(ln->var);
    free(ln->x_hat);
}