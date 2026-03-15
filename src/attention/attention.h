#ifndef ATTENTION_H
#define ATTENTION_H
#include "../../config/model_config.h"
// single head masked self attentino
// input shape: [SEQ_LEN x EMBED_DIM]
// outpup shape: [SEQ_LEN x EMBED_DIM]

typedef struct {
    int embed_dim;
    //projection weights
    float *Wq;
    float *Wk;
    float *Wv;
    //gradients
    float *grad_Wq;
    float *grad_Wk;
    float *grad_Wv;
    // cached tensors for forward and backward pass
    float *Q; // [SEQ_LEN x EMBED_DIM]
    float *K; // [SEQ_LEN x EMBED_DIM]
    float *V; // [SEQ_LEN x EMBED_DIM]
    float *scores;  // [SEQ_LEN x SEQ_LEN]
    float *weights; // [SEQ_LEN x SEQ_LEN]
    float *output; // [SEQ_LEN x EMBED_DIM]
}Attention;

void attention_init(Attention *attn);

void attention_zero_grad(Attention *attn);

void attention_forward(Attention *attn, float *input, float *output);

void attention_backward(Attention *attn, float *input, float *d_output, float *d_input);

void attention_update(Attention *attn, float lr);

void attention_free(Attention *attn);

#endif