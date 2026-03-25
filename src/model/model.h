#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "../../config/model_config.h"
#include "transformer_block.h"
#include "../optim/adam.h"
typedef struct{
   int vocab_size;// [VOCAB_SIZE * EMBED_DIM]
   int embed_dim; // [SEQ_LEN * EMBED_DIM]
   float *pos_embedding;// [SEQ_LEN * EMBED_DIM]
   float *token_embedding;
   float *grad_token_embedding;
   float *embed_buffer; //  [SEQ_LEN * EMBED_DIM]
   TransformerBlock *blocks;
   int num_layers;
   float *block_buffer; // [seq_len *embed_dim]
   float *next_block_buffer; //temporary buffer
   float *block_inputs; // [num_layers * SEQ_LEN * EMBED_DIM] — input to each block saved during forward
   //layernormalization final wala
   LayerNorm final_ln;
   float *ln_out;
   //output layer
   float *W_out; //embed_dim*vocab_size
   float *b_out;
   float *grad_W_out;
   float *grad_b_out;
   float *logits;
   float *d_logits;
   float *d_block;
   AdamOptimizer opt;
   AdamParam emb_opt;
   AdamParam outW_opt;
   AdamParam outb_opt;
   AdamParam adam_fln_g;   // final layer norm gamma
   AdamParam adam_fln_b;   // final layer norm beta
}Model;
void model_init(Model *m);
void model_zero_grad(Model *m);
void model_forward(Model *m, uint16_t *input);
float model_backward(Model *m, uint16_t *input, uint16_t *target);
void model_update(Model *m, float lr);
void model_free(Model *m);

#endif