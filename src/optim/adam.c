#include <stdlib.h>
#include <math.h>
#include "adam.h"

void adam_init_param(AdamParam *p, int size) {
    p->size = size;
    p->m = calloc(size, sizeof(float));
    p->v = calloc(size, sizeof(float));
}

void adam_free_param(AdamParam *p) {
    free(p->m);
    free(p->v);
}

void adam_init(AdamOptimizer *opt, float lr) {
    opt->lr = lr;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->t = 0;
}

void adam_step(AdamOptimizer *opt){
    opt->t += 1;
}

void adam_update(
    AdamOptimizer *opt,
    AdamParam *p,
    float *weights,
    float *grads
) {
    float b1 = opt->beta1;
    float b2 = opt->beta2;

    float b1t = 1.0f - powf(b1, opt->t);
    float b2t = 1.0f - powf(b2, opt->t);

    for (int i = 0; i < p->size; i++) {

        float g = grads[i];

        /* update moments */
        p->m[i] = b1 * p->m[i] + (1.0f - b1) * g;
        p->v[i] = b2 * p->v[i] + (1.0f - b2) * g * g;

        /* bias correction */
        float m_hat = p->m[i] / b1t;
        float v_hat = p->v[i] / b2t;

        /* update weight */
        weights[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
    }
}