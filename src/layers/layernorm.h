#ifndef LAYERNORM_H
#define LAYERNORM_H
#include<stdint.h>
#include "../../config/model_config.h"
// layer normalization layer- operates over embedding dimension D for each timestep t
// input: [SEQ_LEN * EMBED_DIM]
//learnable parameters: gamma and beta, each of shape [EMBED_DIM]
//cached values for backward pass: mean, variance, x_hat (normalized input)
typedef struct{
    int embed_dim;
    float *gamma;
    float *beta;
    float *grad_gamma;
    float *grad_beta;
    float *mean;
    float *var;
    float *x_hat;
}LayerNorm;

void layernorm_init(LayerNorm *ln);
void layernorm_zero_grad(LayerNorm *ln);
// forward pass- input and output are both [SEQ_LEN * EMBED_DIM] 
void layernorm_forward(LayerNorm *ln, float *input, float *output);
// backward pass- input_grad is gradient from next layer and output_grad is gradient w.r.t. input to this layer
void layernorm_backward(LayerNorm *ln, float *input, float *output_grad, float *input_grad);
void layernorm_update(LayerNorm *ln, float lr);
void layernorm_free(LayerNorm *ln);
#endif