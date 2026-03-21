#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "../../config/model_config.h"

#define FF_DIM (4 * EMBED_DIM)

typedef struct {

    int embed_dim;
    int ff_dim;

    /* weights */
    float *W1; // [D × FF]
    float *b1; // [FF]

    float *W2; // [FF × D]
    float *b2; // [D]

    /* gradients */
    float *grad_W1;
    float *grad_b1;

    float *grad_W2;
    float *grad_b2;

    /* buffers */
    float *hidden;   // [SEQ_LEN × FF_DIM]
    float *hidden_act; // after ReLU

} FeedForward;

void feedforward_init(FeedForward *ff);
void feedforward_zero_grad(FeedForward *ff);

void feedforward_forward(
    FeedForward *ff,
    float *input,
    float *output
);

void feedforward_backward(
    FeedForward *ff,
    float *input,
    float *d_output,
    float *d_input
);

void feedforward_update(FeedForward *ff, float lr);
void feedforward_free(FeedForward *ff);

#endif