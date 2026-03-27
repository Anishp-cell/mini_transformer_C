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
#define EMBED_DIM 128     // Scaled up for better capacity (i7 CPU)
#define SEQ_LEN 64        // Sequence Length

/* ================================
   Training Hyperparameters
================================ */
#define LEARNING_RATE 0.0003f
#define TRAIN_STEPS 30000     // Enough steps to learn character mapping
#define PRINT_EVERY 500       // Print less frequently

/* ================================
   Random Initialization
================================ */
#define INIT_STD 0.02f        // Weight initialization std

/* ================================
   Numerical Stability
================================ */
#define EPS 1e-5f

#endif