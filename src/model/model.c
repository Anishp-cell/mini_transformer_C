#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include "model.h"
#include "../layers/layernorm.h"
#include "../attention/attention.h"
#include "../layers/feedforward.h"

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

    //    Allocate embedding

    m->embedding.weight = (float*)malloc(V * D * sizeof(float));
    m->embedding.grad_weight = (float*)malloc(V * D * sizeof(float));

  
    //    Allocate output layer
  
    m->output.W = (float*)malloc(D * V * sizeof(float));
    m->output.b = (float*)malloc(V * sizeof(float));
    m->output.grad_W = (float*)malloc(D * V * sizeof(float));
    m->output.grad_b = (float*)malloc(V * sizeof(float));

    //    Allocate forward buffers

    m->embed_buffer =(float*)malloc(SEQ_LEN * D * sizeof(float));
    m->logit_buffer =(float*)malloc(SEQ_LEN * V * sizeof(float));
    
    m->ln1_buffer= (float*)malloc(SEQ_LEN*D*sizeof(float));
    m->attn_buffer= (float*)malloc(SEQ_LEN*D*sizeof(float));
    m->residual_buffer= (float*)malloc(SEQ_LEN*D*sizeof(float));
    m->ln2_buffer= (float*)malloc(SEQ_LEN*D*sizeof(float));
    
    //    Allocate backward buffer
    m->d_embed_buffer =(float*)malloc(SEQ_LEN * D * sizeof(float));
    m->d_ln1_buffer= (float*)malloc(SEQ_LEN*EMBED_DIM*sizeof(float));
    m->d_attn_buffer= (float*)malloc(SEQ_LEN*EMBED_DIM*sizeof(float));
    m->d_residual_buffer= (float*)malloc(SEQ_LEN*EMBED_DIM*sizeof(float));
    m->d_ln2_buffer= (float*)malloc(SEQ_LEN*EMBED_DIM*sizeof(float));

    //    Initialize layer norm
    layernorm_init(&m->ln1);
    m->attn.embed_dim=D;
    attention_init(&m->attn);
    layernorm_init(&m->ln2);
    //    Initialize weights
    for(int i = 0; i < V * D; i++)
        m->embedding.weight[i] = ran_normal() * INIT_STD;

    for(int i = 0; i < D * V; i++)
        m->output.W[i] = ran_normal() * INIT_STD;

    for(int i = 0; i < V; i++)
        m->output.b[i] = 0.0f;

    feedforward_init(&m->ff);
    m->ff_buffer = malloc(SEQ_LEN * D * sizeof(float));
    m->d_ff_buffer = malloc(SEQ_LEN * D * sizeof(float));
    printf("Model initialized.\n");
}
// zero out gradients before backprop
    void model_zero_grad(Model *m){
        int V= m->vocab_size;
        int D= m->embed_dim;
        memset(m->embedding.grad_weight,0,V*D*sizeof(float));
        memset(m->output.grad_W,0,D*V*sizeof(float));
        memset(m->output.grad_b,0,V*sizeof(float));
        layernorm_zero_grad(&m->ln1);
        layernorm_zero_grad(&m->ln2);
        attention_zero_grad(&m->attn);
        feedforward_zero_grad(&m->ff);

    }
//forward pass
void model_forward(Model *m, uint16_t *input_tokens){
    int V= m->vocab_size;
    int D= m->embed_dim;
    //embedding lookup and linear projection to get logits
    for(int t=0;t<SEQ_LEN;t++){
        uint16_t token=input_tokens[t];
        for(int d=0;d<D;d++){
            m->embed_buffer[t*D+d]= m->embedding.weight[token*D+d];
        }
    }
    layernorm_forward(&m->ln1, m->embed_buffer, m->ln1_buffer); //ln1
    attention_forward(&m->attn, m->ln1_buffer, m->attn_buffer); //attention
    // residual connection before second layer norm (residual 1)
    for(int i = 0; i < SEQ_LEN * D; i++)    {
        m->residual_buffer[i] = m->embed_buffer[i]+m->attn_buffer[i];
    }
    layernorm_forward(&m->ln2, m->residual_buffer, m->ln2_buffer); //ln2
    feedforward_forward(&m->ff,m->ln2_buffer,m->ff_buffer);//feedforward
    for(int i=0;i<SEQ_LEN*D;i++){  //residual 2
        m->ff_buffer[i] += m->residual_buffer[i];
    }
    for(int t=0;t<SEQ_LEN;t++){  //linear
        for(int v=0;v<V;v++){
            float sum= m->output.b[v];
            for(int d= 0;d<D;d++){
                sum += m->ff_buffer[t*D+d] * m->output.W[d*V+v];
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
    memset(m->d_ff_buffer, 0, SEQ_LEN * D * sizeof(float));
    memset(m->d_residual_buffer, 0, SEQ_LEN * D * sizeof(float));
    memset(m->d_ln2_buffer, 0, SEQ_LEN * D * sizeof(float));
    memset(m->d_ln1_buffer, 0, SEQ_LEN * D * sizeof(float));

    for (int t = 0; t < SEQ_LEN; t++) {

        float *logits = &m->logit_buffer[t * V];

        // Softmax
        float max_val = logits[0];
        for (int v = 1; v < V; v++)
            if (logits[v] > max_val) max_val = logits[v];

        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) {
            float val = logits[v] - max_val;
            if(val > 20.0f) val = 20.0f;
            if(val < -20.0f) val = -20.0f;
            logits[v] = expf(val);
            sum_exp += logits[v];
        }
        if(sum_exp < 1e-9f) sum_exp = 1e-9f; 
        for (int v = 0; v < V; v++)
            logits[v] /= sum_exp;

        // Cross-entropy loss
        uint16_t target = target_tokens[t];
        float prob = logits[target];
        // if(prob < 1e-9f) prob = 1e-9f;
        total_loss += -logf(prob + EPS);

        //Gradient of softmax+CE
        logits[target] -= 1.0f;  // dL/dz = p - y

        // Backprop through linear 
        for (int d = 0; d < D; d++) {

            float grad_embed = 0.0f;

            for (int v = 0; v < V; v++) {

                float dz = logits[v];

                m->output.grad_W[d * V + v] +=
                // m->embed_buffer[t * D + d] * dz;
                m->ff_buffer[t * D + d] * dz;
                m->output.grad_b[v] += dz;

                grad_embed += dz * m->output.W[d * V + v];
            }

            // m->d_embed_buffer[t * D + d] = grad_embed;
            m->d_residual_buffer[t * D + d] = grad_embed;
        }
    }
    /* ── Residual-2 Split ─────────────────────────────────────────────────
       The FORWARD pass did:  ff_buffer[i] += residual_buffer[i]
       That addition has TWO input branches, so the gradient fans out equally
       to both branches (chain rule: d(a+b)/da = 1, d(a+b)/db = 1).

       Branch A (FF path):   grad → copy into d_ff_buffer → LN2 → FFN
       Branch B (identity):  grad stays in d_residual_buffer untouched
                             and will be merged back after FFN backward.    */
    for(int i = 0; i < SEQ_LEN * D; i++){
        /* Give the FF branch its own independent copy of the upstream gradient.
           We use d_ff_buffer as scratch so Branch B (d_residual_buffer) is
           preserved unchanged for the identity path merge later.            */
        m->d_ff_buffer[i] = m->d_residual_buffer[i];
    }

    /* ── LN2 Backward ─────────────────────────────────────────────────────
       FORWARD:  ln2_buffer  = LN2(residual_buffer)
       BACKWARD: upstream gradient = d_ff_buffer  (the FF branch copy)
                 output gradient   = d_ln2_buffer  (what FFN will consume)  */
    layernorm_backward(&m->ln2, m->residual_buffer, m->d_ff_buffer, m->d_ln2_buffer);

    /* ── FFN Backward ──────────────────────────────────────────────────────
       FORWARD:  ff_buffer = FFN(ln2_buffer)
       BACKWARD: upstream gradient = d_ln2_buffer (output of LN2 backward)
                 output gradient   = d_ff_buffer  (reused as scratch;
                                     holds dL/d_ln2_input for the merge)    */
    feedforward_backward(&m->ff, m->ln2_buffer, m->d_ln2_buffer, m->d_ff_buffer);
    for(int i = 0; i < SEQ_LEN * D; i++){
        m->d_residual_buffer[i] += m->d_ff_buffer[i];   // merge FF branch gradient back
    }
    attention_backward(&m->attn, m->ln1_buffer, m->d_residual_buffer, m->d_ln1_buffer);
    for(int i=0;i<SEQ_LEN*D;i++){
        m->d_embed_buffer[i] += m->d_residual_buffer[i];
    }
    layernorm_backward(&m->ln1,m->embed_buffer,m->d_ln1_buffer, m->d_embed_buffer);
    for (int t = 0; t < SEQ_LEN; t++) {
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
    int F = m->ff.ff_dim;
    float clip = 1.0f;

    /* ── Helper macro to clip a single value ── */
    #define CLIP(x) ((x) > clip ? clip : ((x) < -clip ? -clip : (x)))

    /* Embedding */
    for(int i = 0; i < V * D; i++){
        m->embedding.grad_weight[i] = CLIP(m->embedding.grad_weight[i]);
        m->embedding.weight[i] -= lr * m->embedding.grad_weight[i];
    }

    /* Output linear */
    for(int i = 0; i < D * V; i++){
        m->output.grad_W[i] = CLIP(m->output.grad_W[i]);
        m->output.W[i] -= lr * m->output.grad_W[i];
    }
    for(int i = 0; i < V; i++){
        m->output.grad_b[i] = CLIP(m->output.grad_b[i]);
        m->output.b[i] -= lr * m->output.grad_b[i];
    }

    /* Attention Wq, Wk, Wv */
    for(int i = 0; i < D * D; i++){
        m->attn.grad_Wq[i] = CLIP(m->attn.grad_Wq[i]);
        m->attn.grad_Wk[i] = CLIP(m->attn.grad_Wk[i]);
        m->attn.grad_Wv[i] = CLIP(m->attn.grad_Wv[i]);
        m->attn.Wq[i] -= lr * m->attn.grad_Wq[i];
        m->attn.Wk[i] -= lr * m->attn.grad_Wk[i];
        m->attn.Wv[i] -= lr * m->attn.grad_Wv[i];
    }

    /* LayerNorm 1 */
    for(int i = 0; i < D; i++){
        m->ln1.grad_gamma[i] = CLIP(m->ln1.grad_gamma[i]);
        m->ln1.grad_beta[i]  = CLIP(m->ln1.grad_beta[i]);
        m->ln1.gamma[i] -= lr * m->ln1.grad_gamma[i];
        m->ln1.beta[i]  -= lr * m->ln1.grad_beta[i];
    }

    /* LayerNorm 2 */
    for(int i = 0; i < D; i++){
        m->ln2.grad_gamma[i] = CLIP(m->ln2.grad_gamma[i]);
        m->ln2.grad_beta[i]  = CLIP(m->ln2.grad_beta[i]);
        m->ln2.gamma[i] -= lr * m->ln2.grad_gamma[i];
        m->ln2.beta[i]  -= lr * m->ln2.grad_beta[i];
    }

    /* FeedForward W1, b1, W2, b2 */
    for(int i = 0; i < D * F; i++){
        m->ff.grad_W1[i] = CLIP(m->ff.grad_W1[i]);
        m->ff.grad_W2[i] = CLIP(m->ff.grad_W2[i]);
        m->ff.W1[i] -= lr * m->ff.grad_W1[i];
        m->ff.W2[i] -= lr * m->ff.grad_W2[i];
    }
    for(int i = 0; i < F; i++){
        m->ff.grad_b1[i] = CLIP(m->ff.grad_b1[i]);
        m->ff.b1[i] -= lr * m->ff.grad_b1[i];
    }
    for(int i = 0; i < D; i++){
        m->ff.grad_b2[i] = CLIP(m->ff.grad_b2[i]);
        m->ff.b2[i] -= lr * m->ff.grad_b2[i];
    }

    #undef CLIP
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
    free(m->ln1_buffer);
    free(m->attn_buffer);
    free(m->residual_buffer);
    free(m->ln2_buffer);
    attention_free(&m->attn);
    layernorm_free(&m->ln1);
    layernorm_free(&m->ln2);
    free(m->d_ln1_buffer);
    free(m->d_ln2_buffer);
    free(m->d_attn_buffer);
    free(m->d_residual_buffer);
}