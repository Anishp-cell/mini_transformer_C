#ifndef MODEL_H
#define MODEL_H

#include<stdint.h>
#include "../config/model_config.h"
//    Embedding Layer
//    weight: [VOCAB_SIZE * EMBED_DIM]
//    grad_weight: same size
typedef struct{
    float* weight;
    float* grad_weight;
}Embedding;
//    Linear Layer (Output Projection)
//    W: [EMBED_DIM * VOCAB_SIZE]
//    b: [VOCAB_SIZE]
//    Gradients mirror same shapes
typedef struct{
    float *W;
    float *b;
    float *grad_W;
    float *grad_b;
}Linear;
//    Model Struct (Phase 1)
//    Embedding → Linear
//    Buffers for forward pass stored inside model
typedef struct{
    //model dimensions
    int vocab_size;
    int embed_dim;
    //layers
    Embedding embedding;
    Linear output;
    //fwd buffers
    float *embed_buffer; // [SEQ_LEN * EMBED_DIM]
    float *logit_buffer; // [SEQ_LEN * VOCAB_SIZE]
    //baclwd buffers
    float *d_embed_buffer; // [SEQ_LEN * EMBED_DIM]
}Model;
void model_init(Model *m);
void model_zero_grad(Model *m);
void model_forward(Model *m, uint16_t *input_tokens);
void model_backward(Model *m, uint16_t *input_tokens,
                    uint16_t *target_tokens);
void model_update(Model *m, float lr);
void model_free(Model *m);
#endif