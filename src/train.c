#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "model/model.h"
// #include "config/model_config.h"

#define TRAIN_DATA_PATH "data/processed/train.bin"

int main() {

    srand(time(NULL));

    /* =====================================================
       Load train.bin
    ===================================================== */
    FILE *file = fopen(TRAIN_DATA_PATH, "rb");
    if (file == NULL) {
        printf("Error: Could not open train.bin\n");
        return 1;
    }

    int total_tokens;
    fread(&total_tokens, sizeof(int), 1, file);

    uint16_t *data = malloc(total_tokens * sizeof(uint16_t));
    if (data == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    fread(data, sizeof(uint16_t), total_tokens, file);
    fclose(file);

    printf("Loaded dataset with %d tokens.\n", total_tokens);

    /* =====================================================
       Initialize Model
    ===================================================== */
    Model model;
    model_init(&model);

    /* =====================================================
       Allocate sequence buffers
    ===================================================== */
    uint16_t input[SEQ_LEN];
    uint16_t target[SEQ_LEN];

    /* =====================================================
       Training Loop
    ===================================================== */
    for (int step = 0; step < TRAIN_STEPS; step++) {

        /* Random starting index */
        int start = rand() % (total_tokens - SEQ_LEN - 1);

        /* Prepare input and target */
        for (int i = 0; i < SEQ_LEN; i++) {
            input[i]  = data[start + i];
            target[i] = data[start + i + 1];
        }

        /* Zero gradients */
        model_zero_grad(&model);

        /* Forward pass */
        model_forward(&model, input);

        /* Backward pass + compute loss */
        float loss = model_backward(&model, input, target);

        /* Update parameters */
        model_update(&model, LEARNING_RATE);

        /* Print loss */
        if (step % PRINT_EVERY == 0) {
            printf("Step %d | Loss: %.4f\n", step, loss);
        }
    }

    /* =====================================================
       Cleanup
    ===================================================== */
    model_free(&model);
    free(data);

    printf("Training complete.\n");

    return 0;
}