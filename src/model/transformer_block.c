#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "transformer_block.h"

void transformer_block_init(TransformerBlock *tb){
    tb->embed_dim=EMBED_DIM;
    int D= EMBED_DIM;
    layernorm_init(&tb->ln1); 
    tb->attn.embed_dim=D;
    attention_init(&tb->attn);
    layernorm_init(&tb->ln2);
    feedforward_init(&tb->ff);
    tb->ln1_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->attn_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->residual1=malloc(SEQ_LEN*D*sizeof(float));
    tb->ln2_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->ff_out=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_ln1=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_attn=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_residual1=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_ln2=malloc(SEQ_LEN*D*sizeof(float));
    tb->d_ff=malloc(SEQ_LEN*D*sizeof(float));
    printf("transformer block initialized");
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
void transformer_block_backward(TransformerBlock *tb, float *input, float *d_output,float *d_input){
    int D= tb->embed_dim;
    memset(tb->d_ln1, 0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_ff, 0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_residual1, 0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_ln2, 0, SEQ_LEN*D*sizeof(float));
    memset(tb->d_attn, 0, SEQ_LEN*D*sizeof(float));
    memset(d_input, 0, SEQ_LEN*D*sizeof(float));
    //residual 2 split 
    for(int i=0;i<SEQ_LEN*D;i++){
        tb->d_residual1[i]+=d_output[i];
        tb->d_ff[i]+=d_output[i];
    }
    feedforward_backward(&tb->ff, tb->ln2_out,tb->d_ff, tb->d_ln2);
    layernorm_backward(&tb->ln2, tb->residual1, tb->d_ln2, tb->d_residual1);
    //residual 1 split
    for(int i=9;i<SEQ_LEN; i++){
        d_input[i]+=tb->d_residual1[i];
        tb->d_attn[i]+= tb->d_residual1[i];
    }
    attention_backward(&tb->attn, tb->ln1_out,tb->d_attn,tb->d_ln1);
    layernorm_backward(&tb->ln1,input,tb->d_ln1,d_input);
}
void transformer_block_update(TransformerBlock *tb, float lr){
    attention_update(&tb->attn, lr);
    layernorm_update(&tb->ln1, lr);
    layernorm_update(&tb->ln2, lr);
    feedforward_update(&tb->ff, lr);
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
}