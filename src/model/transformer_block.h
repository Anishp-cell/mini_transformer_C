#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../layers/layernorm.h"
#include "../attention/attention.h"
#include "../layers/feedforward.h"

typedef struct {

    int embed_dim;

    LayerNorm ln1;
    Attention attn;
    LayerNorm ln2;
    FeedForward ff;

    // forward buffers
    float *ln1_out;
    float *attn_out;
    float *residual1;
    float *ln2_out;
    float *ff_out;

    // backward buffers
    float *d_ln1;
    float *d_attn;
    float *d_residual1;
    float *d_ln2;
    float *d_ff;

} TransformerBlock;

void transformer_block_init(TransformerBlock *tb);
void transformer_block_zero_grad(TransformerBlock *tb);

void transformer_block_forward(
    TransformerBlock *tb,
    float *input,
    float *output
);

void transformer_block_backward(
    TransformerBlock *tb,
    float *input,
    float *d_output,
    float *d_input
);

void transformer_block_update(TransformerBlock *tb, float lr);
void transformer_block_free(TransformerBlock *tb);

#endif