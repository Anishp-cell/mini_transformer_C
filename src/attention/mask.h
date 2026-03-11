#ifndef MASK_H
#define MASK_H

#include "../../config/model_config.h"

/*
    Apply causal mask to attention score matrix.

    scores shape:
        [SEQ_LEN × SEQ_LEN]

    After masking:
        scores[i][j] = -inf  if j > i
*/

static inline void apply_causal_mask(float *scores)
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = i + 1; j < SEQ_LEN; j++)
        {
            scores[i * SEQ_LEN + j] = -1e9f;
        }
    }
}

#endif  