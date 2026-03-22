#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "model.h"

// box muller transform for normal distribution sampling
static float ran_normal(){
    float u1= (rand()+1.0f)/(RAND_MAX+1.0f);
    float u2= (rand()+1.0f)/(RAND_MAX+1.0f);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*M_PI*u2);

}

//model initialization
void model_init(Model *m){

    m->vocab_size = VOCAB_SIZE;
    m->embed_dim=EMBED_DIM;
    m->num_layers=3;
    int V= VOCAB_SIZE;
    int D= EMBED_DIM;
    //set embeddings
    m->token_embedding= (float*)malloc(V*D*sizeof(float));
    m->pos_embedding=(float*)malloc(SEQ_LEN*D*sizeof(float));
    m->embed_buffer=(float*)malloc(SEQ_LEN*D*sizeof(float));
    for(int i=0;i<V*D;i++)
        m->token_embedding[i]=ran_normal()*INIT_STD;
    for(int i=0;i<SEQ_LEN;i++){
        m->pos_embedding[i]=ran_normal()*INIT_STD;
    }
    //initializing number of layers
    m->blocks= (TransformerBlock*)malloc(m->num_layers*sizeof(TransformerBlock));
    for(int i=0;i<m->num_layers;i++)
        transformer_block_init(&m->blocks[i]);
    m->block_buffer=malloc(SEQ_LEN*D*sizeof(float));
    m->next_block_buffer=malloc(SEQ_LEN*D*sizeof(float));
    layernorm_init(&m->final_ln);
    m->ln_out=malloc(SEQ_LEN*D*sizeof(float));
    m->W_out=malloc(D*V*sizeof(float));
    m->b_out=malloc(V*sizeof(float));
    m->grad_W_out=malloc(D*V*sizeof(float));
    m->grad_b_out=malloc(V*sizeof(float));
    for(int i=0;i<D*V;i++)
        m->W_out[i]=ran_normal()*INIT_STD;
    for(int i=0;i<V;i++){
        m->b_out[i]=0.0f;
    }
    m->logits=malloc(SEQ_LEN*V*sizeof(float));
    m->d_logits=malloc(SEQ_LEN*V*sizeof(float));
    m->d_block=malloc(SEQ_LEN*D*sizeof(float));
    printf("Model initialized.\n");
}
// zero out gradients before backprop
    void model_zero_grad(Model *m){
        memset(m->grad_W_out,0,EMBED_DIM*VOCAB_SIZE*sizeof(float));
        memset(m->grad_b_out,0,VOCAB_SIZE*sizeof(float));
        for(int i=0;i<m->num_layers;i++){
            transformer_block_zero_grad(&m->final_ln);
        }
        layernorm_zero_grad(&m->final_ln);
    }
//forward pass
void model_forward(Model *m, uint16_t *input){
    int V= m->vocab_size;
    int D= m->embed_dim;
    //embedding and position embedding
    for(int t=0;t<SEQ_LEN;t++){
        uint16_t token=input[t];
        for(int d=0;d<D;d++){
           float tok=m->token_embedding[token*D+d];
           float pos=m->pos_embedding[t*D+d];
           m->embed_buffer[t*D+d]=tok+pos;
        }
    }
    memcpy(m->block_buffer,m->embed_buffer,SEQ_LEN*D*sizeof(float));
    for(int i=0;i<m->num_layers;i++){
        transformer_block_forward(&m->blocks[i],m->block_buffer, m->next_block_buffer);
         float *tmp = m->block_buffer;
        m->block_buffer = m->next_block_buffer;
        m->next_block_buffer = tmp;
    }
   layernorm_forward(&m->final_ln, m->block_buffer, m->ln_out);

    /* ---- Output ---- */
    for(int t=0;t<SEQ_LEN;t++){
        for(int v=0;v<V;v++){

            float sum = m->b_out[v];

            for(int d=0;d<D;d++){
                sum += m->ln_out[t*D+d] * m->W_out[d*V+v];
            }

            m->logits[t*V+v] = sum;
        }
    }
}

/* =========================
   BACKWARD
========================= */
float model_backward(Model *m,
                     uint16_t *input,
                     uint16_t *target){

    int V = m->vocab_size;
    int D = m->embed_dim;

    float loss = 0.0f;

    memset(m->d_block, 0, SEQ_LEN*D*sizeof(float));

    /* ---- Softmax + CE ---- */
    for(int t=0;t<SEQ_LEN;t++){

        float *logits = &m->logits[t*V];

        float max_val = logits[0];
        for(int v=1;v<V;v++)
            if(logits[v] > max_val) max_val = logits[v];

        float sum = 0.0f;
        for(int v=0;v<V;v++){
            float val = logits[v] - max_val;
            if(val > 20) val = 20;
            if(val < -20) val = -20;

            logits[v] = expf(val);
            sum += logits[v];
        }

        for(int v=0;v<V;v++)
            logits[v] /= sum;

        int y = target[t];
        float p = logits[y];
        if(p < 1e-9f) p = 1e-9f;

        loss += -logf(p);

        logits[y] -= 1.0f;

        /* ---- Output layer backward ---- */
        for(int d=0;d<D;d++){

            float grad = 0.0f;

            for(int v=0;v<V;v++){
                float dz = logits[v];

                m->grad_W_out[d*V+v] +=
                    m->ln_out[t*D+d] * dz;

                m->grad_b_out[v] += dz;

                grad += dz * m->W_out[d*V+v];
            }

            m->d_block[t*D + d] += grad;
        }
    }

    /* ---- Final LN backward ---- */
    layernorm_backward(
        &m->final_ln,
        m->block_buffer,
        m->d_block,
        m->d_block
    );

    /* ---- Blocks backward (reverse) ---- */
    for(int i=m->num_layers-1;i>=0;i--){

        transformer_block_backward(
            &m->blocks[i],
            (i==0) ? m->embed_buffer : m->blocks[i-1].ff_out,
            m->d_block,
            m->d_block
        );
    }

    /* ---- Embedding gradients ---- */
    for(int t=0;t<SEQ_LEN;t++){
        int token = input[t];

        for(int d=0;d<D;d++){
            m->token_embedding[token*D+d] -=
                LEARNING_RATE * m->d_block[t*D+d];
        }
    }

    return loss / SEQ_LEN;
}

/* =========================
   UPDATE
========================= */
void model_update(Model *m, float lr){

    int V = m->vocab_size;
    int D = m->embed_dim;

    for(int i=0;i<D*V;i++)
        m->W_out[i] -= lr * m->grad_W_out[i];

    for(int i=0;i<V;i++)
        m->b_out[i] -= lr * m->grad_b_out[i];

    for(int i=0;i<m->num_layers;i++)
        transformer_block_update(&m->blocks[i], lr);

    layernorm_update(&m->final_ln, lr);
}

/* =========================
   FREE
========================= */
void model_free(Model *m){

    free(m->token_embedding);
    free(m->pos_embedding);
    free(m->embed_buffer);

    free(m->block_buffer);
    free(m->next_block_buffer);

    for(int i=0;i<m->num_layers;i++)
        transformer_block_free(&m->blocks[i]);

    free(m->blocks);

    free(m->W_out);
    free(m->b_out);
    free(m->grad_W_out);
    free(m->grad_b_out);

    free(m->logits);
    free(m->d_logits);
    free(m->d_block);

    free(m->ln_out);

    layernorm_free(&m->final_ln);
}