#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
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
    m->embed_dim  = EMBED_DIM;

    int V = m->vocab_size;
    int D = m->embed_dim;

    /* =========================
       Allocate embedding
    ========================= */
    m->embedding.weight = (float*)malloc(V * D * sizeof(float));
    m->embedding.grad_weight = (float*)malloc(V * D * sizeof(float));

    /* =========================
       Allocate output layer
    ========================= */
    m->output.W = (float*)malloc(D * V * sizeof(float));
    m->output.b = (float*)malloc(V * sizeof(float));
    m->output.grad_W = (float*)malloc(D * V * sizeof(float));
    m->output.grad_b = (float*)malloc(V * sizeof(float));

    /* =========================
       Allocate forward buffers
    ========================= */
    m->embed_buffer =
        (float*)malloc(SEQ_LEN * D * sizeof(float));

    m->logit_buffer =
        (float*)malloc(SEQ_LEN * V * sizeof(float));

    /* =========================
       Allocate backward buffer
    ========================= */
    m->d_embed_buffer =
        (float*)malloc(SEQ_LEN * D * sizeof(float));

    /* =========================
       Initialize weights
    ========================= */
    for(int i = 0; i < V * D; i++)
        m->embedding.weight[i] = ran_normal() * INIT_STD;

    for(int i = 0; i < D * V; i++)
        m->output.W[i] = ran_normal() * INIT_STD;

    for(int i = 0; i < V; i++)
        m->output.b[i] = 0.0f;

    printf("Model initialized.\n");
}
// zero out gradients before backprop
    void model_zero_grad(Model *m){
        int V= m->vocab_size;;
        int D= m->embed_dim;
        memset(m->embedding.grad_weight,0,V*D*sizeof(float));
        memset(m->output.grad_W,0,D*V*sizeof(float));
        memset(m->output.grad_b,0,V*sizeof(float));
    }
//forward pass
void model_forward(Model *m, uint16_t *input_tokens){
    int V= m->vocab_size;
    int D= m->embed_dim;
    for(int t=0;t<SEQ_LEN;t++){
        uint16_t token=input_tokens[t];
        for(int d=0;d<D;d++){
            m->embed_buffer[t*D+d]= m->embedding.weight[token*D+d];
        }
        for(int v=0;v<V;v++){
            float sum= m->output.b[v];
            for(int d= 0;d<D;d++){
                sum+=m->embed_buffer[t*D+d]*m->output.W[d*V+v];;
            }
            m->logit_buffer[t*V+v]=sum;
        }
    }
}
//backward pass
float model_backward(Model *m,
                     uint16_t *input_tokens,
                     uint16_t *target_tokens) {

    int V = m->vocab_size;
    int D = m->embed_dim;

    float total_loss = 0.0f;

    /* Zero d_embed buffer */
    memset(m->d_embed_buffer, 0, SEQ_LEN * D * sizeof(float));

    for (int t = 0; t < SEQ_LEN; t++) {

        float *logits = &m->logit_buffer[t * V];

        /* ---- Softmax ---- */
        float max_val = logits[0];
        for (int v = 1; v < V; v++)
            if (logits[v] > max_val) max_val = logits[v];

        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) {
            logits[v] = expf(logits[v] - max_val);
            sum_exp += logits[v];
        }

        for (int v = 0; v < V; v++)
            logits[v] /= sum_exp;

        /* ---- Cross-entropy loss ---- */
        uint16_t target = target_tokens[t];
        float prob = logits[target];
        total_loss += -logf(prob + EPS);

        /* ---- Gradient of softmax+CE ---- */
        logits[target] -= 1.0f;  // dL/dz = p - y

        /* ---- Backprop through linear ---- */
        for (int d = 0; d < D; d++) {

            float grad_embed = 0.0f;

            for (int v = 0; v < V; v++) {

                float dz = logits[v];

                m->output.grad_W[d * V + v] +=
                    m->embed_buffer[t * D + d] * dz;

                m->output.grad_b[v] += dz;

                grad_embed += dz * m->output.W[d * V + v];
            }

            m->d_embed_buffer[t * D + d] = grad_embed;
        }

        /* ---- Backprop to embedding ---- */
        uint16_t token = input_tokens[t];

        for (int d = 0; d < D; d++) {
            m->embedding.grad_weight[token * D + d] +=
                m->d_embed_buffer[t * D + d];
        }
    }

    return total_loss / SEQ_LEN;
}

void model_update(Model *m, float lr) {

    int V = m->vocab_size;
    int D = m->embed_dim;

    for (int i = 0; i < V * D; i++)
        m->embedding.weight[i] -= lr * m->embedding.grad_weight[i];

    for (int i = 0; i < D * V; i++)
        m->output.W[i] -= lr * m->output.grad_W[i];

    for (int i = 0; i < V; i++)
        m->output.b[i] -= lr * m->output.grad_b[i];
}
void model_free(Model *m) {

    free(m->embedding.weight);
    free(m->embedding.grad_weight);

    free(m->output.W);
    free(m->output.b);
    free(m->output.grad_W);
    free(m->output.grad_b);

    free(m->embed_buffer);
    free(m->logit_buffer);
    free(m->d_embed_buffer);
}