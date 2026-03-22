#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

/* ================================
   Dataset / Vocabulary
================================ */
#define VOCAB_SIZE 65     // From vocab_builder output

/* ================================
   Model Dimensions (Phase 1)
   Keep small for debugging
================================ */
#define EMBED_DIM 64      // Embedding size
#define SEQ_LEN 64        // Training sequence length

/* ================================
   Training Hyperparameters
================================ */
#define LEARNING_RATE 0.0003f
#define TRAIN_STEPS 2000      // Total update steps
#define PRINT_EVERY 100       // Print loss every N steps

/* ================================
   Random Initialization
================================ */
#define INIT_STD 0.02f        // Weight initialization std

/* ================================
   Numerical Stability
================================ */
#define EPS 1e-5f

#endif