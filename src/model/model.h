#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "../../config/model_config.h"
#include "../layers/layernorm.h"
#include "../attention/attention.h"
#include "../layers/feedforward.h"

/* ============================================================
   Embedding Layer
   weight: [VOCAB_SIZE * EMBED_DIM]
   grad_weight: same size
============================================================ */
typedef struct {
    float *weight;
    float *grad_weight;
} Embedding;

/* ============================================================
   Linear Layer (Output Projection)
   W: [EMBED_DIM * VOCAB_SIZE]
   b: [VOCAB_SIZE]
   Gradients mirror same shapes
============================================================ */
typedef struct {
    float *W;
    float *b;
    float *grad_W;
    float *grad_b;
} Linear;

/* ============================================================
   Model Struct (Phase 1)
   Embedding → Linear
============================================================ */
typedef struct {

    /* Model dimensions */
    int vocab_size;
    int embed_dim;

    /* Layers */
    Embedding embedding;
    Linear output;
    LayerNorm ln1;
    Attention attn;
    LayerNorm ln2;
    /* Forward buffers */
    float *embed_buffer;    // [SEQ_LEN * EMBED_DIM]
    float *logit_buffer;   // [SEQ_LEN * VOCAB_SIZE]
    float *ln1_buffer;      // output of layer norm [SEQ_LEN * EMBED_DIM]
    float *attn_buffer;     // output of attention [SEQ_LEN * EMBED_DIM]
    float *residual_buffer; // output of attention before second layer norm [SEQ_LEN * EMBED_DIM]
    float *ln2_buffer;      // output of layer norm [SEQ_LEN * EMBED_DIM]
    /* Backward buffer */
    float *d_embed_buffer;  // [SEQ_LEN * EMBED_DIM]
    float *d_ln1_buffer;     // [SEQ_LEN * EMBED_DIM] 
    float *d_attn_buffer; 
    float *d_residual_buffer;
    float *d_ln2_buffer;
    FeedForward ff;
   float *ff_buffer;
   float *d_ff_buffer;
    
} Model;

/* ============================================================
   API
============================================================ */
void model_init(Model *m);
void model_zero_grad(Model *m);
void model_forward(Model *m, uint16_t *input_tokens);

/* MUST return float (loss) */
float model_backward(Model *m,
                     uint16_t *input_tokens,
                     uint16_t *target_tokens);

void model_update(Model *m, float lr);
void model_free(Model *m);


#endif