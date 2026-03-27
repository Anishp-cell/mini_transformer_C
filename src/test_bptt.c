#include <stdio.h>
#include <stdlib.h>
#include "model/model.h"

int main() {
    printf("Starting BPTT Correctness Test...\n");
    Model model;
    model_init(&model);

    uint16_t input1[SEQ_LEN];
    uint16_t target1[SEQ_LEN];
    for(int i=0; i<SEQ_LEN; i++) {
        input1[i] = i % VOCAB_SIZE;
        target1[i] = (i + 1) % VOCAB_SIZE;
    }

    // Run 1
    model_zero_grad(&model);
    model_forward(&model, input1);
    float loss1 = model_loss(&model, target1);
    model_backward(&model, input1);
    float grad1_W = model.grad_W_out[0];
    float grad1_tok = model.grad_token_embedding[0];

    // Run 2 (same input)
    model_zero_grad(&model);
    model_forward(&model, input1);
    float loss2 = model_loss(&model, target1);
    model_backward(&model, input1);
    float grad2_W = model.grad_W_out[0];
    float grad2_tok = model.grad_token_embedding[0];

    printf("Run 1 (Input A): Loss = %f, grad_W_out[0] = %f, grad_tok[0] = %f\n", loss1, grad1_W, grad1_tok);
    printf("Run 2 (Input A): Loss = %f, grad_W_out[0] = %f, grad_tok[0] = %f\n", loss2, grad2_W, grad2_tok);

    if (loss1 == loss2 && grad1_W == grad2_W && grad1_tok == grad2_tok) {
         printf("=> SUCCESS: Deterministic BPTT behavior verified.\n");
    } else {
         printf("=> FAILED: BPTT is not deterministic!\n");
    }

    uint16_t input2[SEQ_LEN];
    uint16_t target2[SEQ_LEN];
    for(int i=0; i<SEQ_LEN; i++) {
        input2[i] = (i*2) % VOCAB_SIZE;
        target2[i] = (i*2 + 1) % VOCAB_SIZE;
    }

    // Run 3 (different input)
    model_zero_grad(&model);
    model_forward(&model, input2);
    float loss3 = model_loss(&model, target2);
    model_backward(&model, input2);
    float grad3_W = model.grad_W_out[0];
    float grad3_tok = model.grad_token_embedding[0];

    printf("Run 3 (Input B): Loss = %f, grad_W_out[0] = %f, grad_tok[0] = %f\n", loss3, grad3_W, grad3_tok);
    if (loss1 != loss3 && grad1_W != grad3_W) {
         printf("=> SUCCESS: Different inputs yield different gradients/loss.\n");
    } else {
         printf("=> FAILED: Model is unresponsive to input changes!\n");
    }

    model_free(&model);
    return 0;
}
