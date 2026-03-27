#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "transformer_block.h"

void transformer_block_init(TransformerBlock *tb){
    printf("TB: start\n");
    tb->embed_dim=EMBED_DIM;
    int D= EMBED_DIM;

    printf("TB: ln1 init\n");
    layernorm_init(&tb->ln1); 

    printf("TB: attention init\n");
    tb->attn.embed_dim=D;
    attention_init(&tb->attn);

    printf("TB: ln2 init\n");
    layernorm_init(&tb->ln2);

    printf("TB: ff init\n");
    feedforward_init(&tb->ff);

    printf("TB: buffers alloc\n");
    tb->ln1_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->attn_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->residual1=malloc(SEQ_LEN*D*sizeof(float));
    tb->ln2_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->ff_out=malloc(SEQ_LEN*D*sizeof(float));

    printf("TB: grad buffers alloc\n");
    tb->d_ln1=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_attn=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_residual1=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_ln2=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_ff=malloc(SEQ_LEN*D*sizeof(float));


    int F = FF_DIM;
    adam_init_param(&tb->adam_Wq,    D*D);
    adam_init_param(&tb->adam_Wk,    D*D);
    adam_init_param(&tb->adam_Wv,    D*D);
    adam_init_param(&tb->adam_W1,    D*F);
    adam_init_param(&tb->adam_b1,    F);
    adam_init_param(&tb->adam_W2,    F*D);
    adam_init_param(&tb->adam_b2,    D);
    adam_init_param(&tb->adam_ln1_g, D);
    adam_init_param(&tb->adam_ln1_b, D);
    adam_init_param(&tb->adam_ln2_g, D);
    adam_init_param(&tb->adam_ln2_b, D);
    printf("transformer block initialized\n");
}
void transformer_block_zero_grad(TransformerBlock *tb){
    layernorm_zero_grad(&tb->ln1);
    layernorm_zero_grad(&tb->ln2);
    attention_zero_grad(&tb->attn);
    feedforward_zero_grad(&tb->ff);
}
void transformer_block_forward(TransformerBlock *tb, float *input, float *output){
    int D= tb->embed_dim;
    layernorm_forward(&tb->ln1, input, tb->ln1_out);
    attention_forward(&tb->attn, tb->ln1_out, tb->attn_out);
    //residual 1
    for(int i=0;i<SEQ_LEN*D;i++)
        tb->residual1[i] = input[i] + tb->attn_out[i];
    layernorm_forward(&tb->ln2, tb->residual1, tb->ln2_out);
    feedforward_forward(&tb->ff, tb->ln2_out, tb->ff_out);
    for(int i=0;i<SEQ_LEN*D;i++)
        output[i] = tb->residual1[i] + tb->ff_out[i];

}
void transformer_block_backward(TransformerBlock *tb, float *input, float *d_output, float *d_input){
    int D = tb->embed_dim;
    memset(tb->d_ln1,       0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_ff,        0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_residual1, 0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_ln2,       0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_attn,      0, SEQ_LEN*D*sizeof(float));

    // residual 2 split- we copy d_output to d_ff and d_residual1 so that 
    // gradient can flow to both branches
    for(int i = 0; i < SEQ_LEN*D; i++){
        tb->d_ff[i]        = d_output[i];  
        tb->d_residual1[i] = d_output[i]; 
    }
    // d_input will accumulate gradients below - zero it first 
    memset(d_input, 0, SEQ_LEN*D*sizeof(float));
    // ffn backward
    feedforward_backward(&tb->ff, tb->ln2_out, tb->d_ff, tb->d_ln2);
    // ln2 backward
    layernorm_backward(&tb->ln2, tb->residual1, tb->d_ln2, tb->d_ff);

    // residual 1 merge- here we add the gradient from the ffn path to the 
    // identity path and the attention path
    for(int i = 0; i < SEQ_LEN*D; i++){       
        tb->d_residual1[i] += tb->d_ff[i];    
        d_input[i]         += tb->d_residual1[i]; 
        tb->d_attn[i]      += tb->d_residual1[i]; 
    }

    // attention backward
    attention_backward(&tb->attn, tb->ln1_out, tb->d_attn, tb->d_ln1);

    // ln1 backward
    layernorm_backward(&tb->ln1, input, tb->d_ln1, d_input);
}
void transformer_block_update(TransformerBlock *tb, AdamOptimizer *opt){
    // attention projections
    adam_update(opt, &tb->adam_Wq, tb->attn.Wq, tb->attn.grad_Wq);
    adam_update(opt, &tb->adam_Wk, tb->attn.Wk, tb->attn.grad_Wk);
    adam_update(opt, &tb->adam_Wv, tb->attn.Wv, tb->attn.grad_Wv);
    // feedforward weights and biases
    adam_update(opt, &tb->adam_W1, tb->ff.W1, tb->ff.grad_W1);
    adam_update(opt, &tb->adam_b1, tb->ff.b1, tb->ff.grad_b1);
    adam_update(opt, &tb->adam_W2, tb->ff.W2, tb->ff.grad_W2);
    adam_update(opt, &tb->adam_b2, tb->ff.b2, tb->ff.grad_b2);
    // layer norm 1 gamma and beta
    adam_update(opt, &tb->adam_ln1_g, tb->ln1.gamma, tb->ln1.grad_gamma);
    adam_update(opt, &tb->adam_ln1_b, tb->ln1.beta,  tb->ln1.grad_beta);
    // layer norm 2 gamma and beta
    adam_update(opt, &tb->adam_ln2_g, tb->ln2.gamma, tb->ln2.grad_gamma);
    adam_update(opt, &tb->adam_ln2_b, tb->ln2.beta,  tb->ln2.grad_beta);
}

void transformer_block_free(TransformerBlock *tb){
    free(tb->ln1_out);
    free(tb->attn_out);
    free(tb->residual1);
    free(tb->ln2_out);
    free(tb->ff_out);
    free(tb->d_ln1);
    free(tb->d_attn);
    free(tb->d_residual1);
    free(tb->d_ln2);
    free(tb->d_ff);
    attention_free(&tb->attn);
    layernorm_free(&tb->ln1);
    layernorm_free(&tb->ln2);
    feedforward_free(&tb->ff);
    adam_free_param(&tb->adam_Wq);    adam_free_param(&tb->adam_Wk);    adam_free_param(&tb->adam_Wv);
    adam_free_param(&tb->adam_W1);    adam_free_param(&tb->adam_b1);
    adam_free_param(&tb->adam_W2);    adam_free_param(&tb->adam_b2);
    adam_free_param(&tb->adam_ln1_g); adam_free_param(&tb->adam_ln1_b);
    adam_free_param(&tb->adam_ln2_g); adam_free_param(&tb->adam_ln2_b);
}