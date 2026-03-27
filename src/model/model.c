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

    printf("MI: start\n");

    m->vocab_size = VOCAB_SIZE;
    m->embed_dim=EMBED_DIM;
    m->num_layers=3;
    int V= VOCAB_SIZE;
    int D= EMBED_DIM;

    printf("MI: embeddings alloc\n");
    //set embeddings
    m->token_embedding= (float*)malloc(V*D*sizeof(float));
    m->pos_embedding=(float*)malloc(SEQ_LEN*D*sizeof(float));
    m->embed_buffer=(float*)malloc(SEQ_LEN*D*sizeof(float));

    printf("MI: embeddings init\n");
    for(int i=0;i<V*D;i++)
        m->token_embedding[i]=ran_normal()*INIT_STD;
    for(int i=0;i<SEQ_LEN*D;i++){
        m->pos_embedding[i]=ran_normal()*INIT_STD;
    }
    printf("MI: blocks alloc\n");
    //initializing number of layers
    m->blocks= (TransformerBlock*)malloc(m->num_layers*sizeof(TransformerBlock));

    printf("MI: blocks init loop start\n");
    for(int i=0;i<m->num_layers;i++)
        transformer_block_init(&m->blocks[i]);
    printf("MI: blocks init done\n");

    printf("MI: buffers alloc\n");
    m->block_buffer      = malloc(SEQ_LEN*D*sizeof(float));
    m->next_block_buffer = malloc(SEQ_LEN*D*sizeof(float));

    /* Store the INPUT to each block so backward can use the correct buffer.
       block_inputs[i] = input fed into blocks[i] during forward pass.     */
    m->block_inputs = malloc(m->num_layers * SEQ_LEN * D * sizeof(float));

    printf("MI: final ln init\n");
    layernorm_init(&m->final_ln);

    printf("MI: output alloc\n");
    m->ln_out=malloc(SEQ_LEN*D*sizeof(float));
    m->W_out=malloc(D*V*sizeof(float));
    m->b_out=malloc(V*sizeof(float));
    m->grad_W_out=malloc(D*V*sizeof(float));
    m->grad_b_out=malloc(V*sizeof(float));
    printf("MI: memset grad_token_embedding\n");
    m->grad_token_embedding=malloc(V*D*sizeof(float));
    memset(m->grad_token_embedding, 0, V * D * sizeof(float));
    m->grad_pos_embedding=malloc(SEQ_LEN*D*sizeof(float));
    memset(m->grad_pos_embedding, 0, SEQ_LEN * D * sizeof(float));

    printf("MI: output init\n");
    for(int i=0;i<D*V;i++)
        m->W_out[i]=ran_normal()*INIT_STD;
    for(int i=0;i<V;i++){
        m->b_out[i]=0.0f;
    }

    printf("MI: memset grads\n");
    memset(m->grad_W_out, 0, D * V * sizeof(float));
    memset(m->grad_b_out, 0, V * sizeof(float));

    printf("MI: buffers alloc 2\n");
    m->logits=malloc(SEQ_LEN*V*sizeof(float));
    m->d_logits=malloc(SEQ_LEN*V*sizeof(float));
    m->d_block=malloc(SEQ_LEN*D*sizeof(float));
    m->d_ln_out=malloc(SEQ_LEN*D*sizeof(float));

    printf("MI: adam init\n");
    adam_init(&m->opt, LEARNING_RATE);
    adam_init_param(&m->emb_opt,    V * D);
    adam_init_param(&m->outW_opt,   D * V);
    adam_init_param(&m->outb_opt,   V);
    adam_init_param(&m->adam_fln_g, D);
    adam_init_param(&m->adam_fln_b, D);
    adam_init_param(&m->adam_pos_opt, SEQ_LEN * D);

    printf("Model initialized.\n");
}
/* Zero out ALL gradients before backward — every parameter group */
void model_zero_grad(Model *m){
    memset(m->grad_W_out,          0, EMBED_DIM*VOCAB_SIZE*sizeof(float));
    memset(m->grad_b_out,          0, VOCAB_SIZE*sizeof(float));
    /* BUG FIX: token embedding grad must be zeroed each step too */
    memset(m->grad_token_embedding, 0, VOCAB_SIZE*EMBED_DIM*sizeof(float));
    memset(m->grad_pos_embedding,   0, SEQ_LEN*EMBED_DIM*sizeof(float));
    /* BUG FIX: zero each transformer block's internal gradients */
    for(int i = 0; i < m->num_layers; i++)
        transformer_block_zero_grad(&m->blocks[i]);
    /* zero the final layer norm gradients */
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
    memcpy(m->block_buffer, m->embed_buffer, SEQ_LEN*D*sizeof(float));
    for(int i = 0; i < m->num_layers; i++){
        memcpy(&m->block_inputs[i * SEQ_LEN * D],
               m->block_buffer,
               SEQ_LEN * D * sizeof(float));
        transformer_block_forward(&m->blocks[i], m->block_buffer, m->next_block_buffer);
        float *tmp       = m->block_buffer;
        m->block_buffer  = m->next_block_buffer;
        m->next_block_buffer = tmp;
    }
   layernorm_forward(&m->final_ln, m->block_buffer, m->ln_out);

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

float model_loss(Model *m, uint16_t *target) {
    int V = m->vocab_size;
    float loss = 0.0f;

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
            logits[v] /= (sum < 1e-9f ? 1e-9f : sum);

        int y = target[t];
        float p = logits[y];
        if(p < 1e-9f) p = 1e-9f;

        loss += -logf(p);

        for(int v=0; v<V; v++){
            m->d_logits[t*V + v] = logits[v];
        }
        m->d_logits[t*V + y] -= 1.0f;
    }
    return loss / SEQ_LEN;
}

void model_backward(Model *m, uint16_t *input){

    int V = m->vocab_size;
    int D = m->embed_dim;

    memset(m->d_block, 0, SEQ_LEN*D*sizeof(float));

    for(int t=0;t<SEQ_LEN;t++){

        for(int d=0;d<D;d++){

            float grad = 0.0f;

            for(int v=0;v<V;v++){
                float dz = m->d_logits[t*V + v];;

                m->grad_W_out[d*V+v] +=
                    m->ln_out[t*D+d] * dz;

                m->grad_b_out[v] += dz;

                grad += dz * m->W_out[d*V+v];
            }

            m->d_block[t*D + d] += grad;
        }
    }
    memset(m->d_ln_out, 0, SEQ_LEN * EMBED_DIM * sizeof(float));
    layernorm_backward(
        &m->final_ln,
        m->block_buffer,  
        m->d_block,       
        m->d_ln_out       
    );
    memcpy(m->d_block, m->d_ln_out, SEQ_LEN * EMBED_DIM * sizeof(float));

    for(int i = m->num_layers-1; i >= 0; i--){
        float *block_input = &m->block_inputs[i * SEQ_LEN * EMBED_DIM];
        transformer_block_backward(
            &m->blocks[i],
            block_input,  
            m->d_block,   
            m->d_block    
        );
    }

    /* Token + position embedding gradients
       Both token_embedding[token] and pos_embedding[t] fed into embed_buffer equally,
       so dL/d(pos_embedding[t]) = dL/d(embed_buffer[t]) = same as token grad.      */
    for(int t = 0; t < SEQ_LEN; t++){
        int token = input[t];
        for(int d = 0; d < EMBED_DIM; d++){
            float g = m->d_block[t * EMBED_DIM + d];
            m->grad_token_embedding[token * EMBED_DIM + d] += g;
            m->grad_pos_embedding[t * EMBED_DIM + d]       += g;
        }
    }

}


/* Clip global gradient norm to max_norm.
   Computes L2 norm over ALL gradients in the model,
   then scales all down uniformly if norm > max_norm.  */
void model_clip_grad_norm(Model *m, float max_norm) {
    int V = m->vocab_size, D = m->embed_dim;
    float norm_sq = 0.0f;

    // token + pos embeddings
    for(int i = 0; i < V*D;       i++) norm_sq += m->grad_token_embedding[i] * m->grad_token_embedding[i];
    for(int i = 0; i < SEQ_LEN*D; i++) norm_sq += m->grad_pos_embedding[i]   * m->grad_pos_embedding[i];
    // output layer
    for(int i = 0; i < D*V; i++) norm_sq += m->grad_W_out[i] * m->grad_W_out[i];
    for(int i = 0; i < V;   i++) norm_sq += m->grad_b_out[i] * m->grad_b_out[i];
    // final layer norm
    for(int i = 0; i < D; i++) norm_sq += m->final_ln.grad_gamma[i] * m->final_ln.grad_gamma[i];
    for(int i = 0; i < D; i++) norm_sq += m->final_ln.grad_beta[i]  * m->final_ln.grad_beta[i];
    // transformer blocks
    for(int b = 0; b < m->num_layers; b++){
        TransformerBlock *tb = &m->blocks[b];
        int F = tb->ff.ff_dim;
        for(int i=0;i<D*D;i++){ norm_sq+=tb->attn.grad_Wq[i]*tb->attn.grad_Wq[i]; }
        for(int i=0;i<D*D;i++){ norm_sq+=tb->attn.grad_Wk[i]*tb->attn.grad_Wk[i]; }
        for(int i=0;i<D*D;i++){ norm_sq+=tb->attn.grad_Wv[i]*tb->attn.grad_Wv[i]; }
        for(int i=0;i<D*F;i++){ norm_sq+=tb->ff.grad_W1[i]*tb->ff.grad_W1[i]; }
        for(int i=0;i<F*D;i++){ norm_sq+=tb->ff.grad_W2[i]*tb->ff.grad_W2[i]; }
        for(int i=0;i<F;  i++){ norm_sq+=tb->ff.grad_b1[i]*tb->ff.grad_b1[i]; }
        for(int i=0;i<D;  i++){ norm_sq+=tb->ff.grad_b2[i]*tb->ff.grad_b2[i]; }
        for(int i=0;i<D;  i++){ norm_sq+=tb->ln1.grad_gamma[i]*tb->ln1.grad_gamma[i]; }
        for(int i=0;i<D;  i++){ norm_sq+=tb->ln1.grad_beta[i] *tb->ln1.grad_beta[i];  }
        for(int i=0;i<D;  i++){ norm_sq+=tb->ln2.grad_gamma[i]*tb->ln2.grad_gamma[i]; }
        for(int i=0;i<D;  i++){ norm_sq+=tb->ln2.grad_beta[i] *tb->ln2.grad_beta[i];  }
    }

    float norm = sqrtf(norm_sq);
    if(norm <= max_norm) return;   // no clipping needed

    float scale = max_norm / norm; // scale all grads by this factor

    for(int i=0;i<V*D;      i++) m->grad_token_embedding[i] *= scale;
    for(int i=0;i<SEQ_LEN*D;i++) m->grad_pos_embedding[i]   *= scale;
    for(int i=0;i<D*V;i++) m->grad_W_out[i] *= scale;
    for(int i=0;i<V;  i++) m->grad_b_out[i] *= scale;
    for(int i=0;i<D;  i++){ m->final_ln.grad_gamma[i]*=scale; m->final_ln.grad_beta[i]*=scale; }

    for(int b=0;b<m->num_layers;b++){
        TransformerBlock *tb = &m->blocks[b];
        int F = tb->ff.ff_dim;
        for(int i=0;i<D*D;i++){ tb->attn.grad_Wq[i]*=scale; tb->attn.grad_Wk[i]*=scale; tb->attn.grad_Wv[i]*=scale; }
        for(int i=0;i<D*F;i++) tb->ff.grad_W1[i]*=scale;
        for(int i=0;i<F*D;i++) tb->ff.grad_W2[i]*=scale;
        for(int i=0;i<F;  i++) tb->ff.grad_b1[i]*=scale;
        for(int i=0;i<D;  i++) tb->ff.grad_b2[i]*=scale;
        for(int i=0;i<D;  i++){ tb->ln1.grad_gamma[i]*=scale; tb->ln1.grad_beta[i]*=scale; }
        for(int i=0;i<D;  i++){ tb->ln2.grad_gamma[i]*=scale; tb->ln2.grad_beta[i]*=scale; }
    }
}

void model_update(Model *m, float lr){

    int V = m->vocab_size;
    int D = m->embed_dim;

    adam_step(&m->opt);

    // Token embedding
    adam_update(&m->opt, &m->emb_opt,     m->token_embedding, m->grad_token_embedding);
    // Positional embedding
    adam_update(&m->opt, &m->adam_pos_opt, m->pos_embedding,  m->grad_pos_embedding);
    // Output weight 
    adam_update(&m->opt, &m->outW_opt, m->W_out,           m->grad_W_out);
    // Output bias
    adam_update(&m->opt, &m->outb_opt, m->b_out,           m->grad_b_out);

    for(int i = 0; i < m->num_layers; i++)
        transformer_block_update(&m->blocks[i], &m->opt);

    // final layer norm — also updated with Adam
    adam_update(&m->opt, &m->adam_fln_g, m->final_ln.gamma, m->final_ln.grad_gamma);
    adam_update(&m->opt, &m->adam_fln_b, m->final_ln.beta,  m->final_ln.grad_beta);
}

void model_free(Model *m){

    free(m->token_embedding);
    free(m->pos_embedding);
    free(m->grad_token_embedding);
    free(m->grad_pos_embedding);
    free(m->embed_buffer);

    free(m->block_buffer);
    free(m->next_block_buffer);
    free(m->block_inputs); 

    for(int i=0;i<m->num_layers;i++)
        transformer_block_free(&m->blocks[i]);

    free(m->blocks);

    free(m->W_out);
    free(m->b_out);
    free(m->grad_W_out);
    free(m->grad_b_out);
    free(m->grad_token_embedding);

    free(m->logits);
    free(m->d_logits);
    free(m->d_block);

    free(m->ln_out);
    free(m->d_ln_out);

    layernorm_free(&m->final_ln);

    adam_free_param(&m->emb_opt);
    adam_free_param(&m->outW_opt);
    adam_free_param(&m->outb_opt);
    adam_free_param(&m->adam_fln_g);
    adam_free_param(&m->adam_fln_b);
    adam_free_param(&m->adam_pos_opt);
}