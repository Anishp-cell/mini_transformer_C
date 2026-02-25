#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define INPUT_PATH  "../data/raw/input.txt"
#define VOCAB_PATH  "../data/processed/vocab.bin"
#define OUTPUT_PATH "../data/processed/train.bin"

int main() {

    /* -----------------------------
       Load vocab.bin
    ----------------------------- */
    FILE *vocab_file = fopen(VOCAB_PATH, "rb");
    if (vocab_file == NULL) {
        printf("Error: Could not open vocab.bin\n");
        return 1;
    }

    int vocab_size;
    fread(&vocab_size, sizeof(int), 1, vocab_file);

    unsigned char *id_to_char = malloc(vocab_size * sizeof(unsigned char));
    if (id_to_char == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    fread(id_to_char, sizeof(unsigned char), vocab_size, vocab_file);
    fclose(vocab_file);

    int char_to_id[256];
    for (int i = 0; i < 256; i++) {
        char_to_id[i] = -1;
    }

    for (int i = 0; i < vocab_size; i++) {
        char_to_id[id_to_char[i]] = i;
    }

    printf("Loaded vocab. Size = %d\n", vocab_size);

    /* -----------------------------
       Open input.txt
    ----------------------------- */
    FILE *input_file = fopen(INPUT_PATH, "rb");
    if (input_file == NULL) {
        printf("Error: Could not open input.txt\n");
        return 1;
    }

    /* Count total tokens */
    int c;
    int total_tokens = 0;

    while ((c = fgetc(input_file)) != EOF) {
        total_tokens++;
    }

    if (total_tokens == 0) {
        printf("Input file is empty.\n");
        return 1;
    }

    printf("Total tokens: %d\n", total_tokens);

    fseek(input_file, 0, SEEK_SET);

    /* Allocate token buffer */
    uint16_t *tokens = malloc(total_tokens * sizeof(uint16_t));
    if (tokens == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    /* Convert characters to token IDs */
    int index = 0;

    while ((c = fgetc(input_file)) != EOF) {

        int token_id = char_to_id[(unsigned char)c];

        if (token_id == -1) {
            printf("Error: Character not found in vocab.\n");
            return 1;
        }

        tokens[index++] = (uint16_t)token_id;
    }

    fclose(input_file);

    /* -----------------------------
       Save train.bin
    ----------------------------- */
    FILE *output_file = fopen(OUTPUT_PATH, "wb");
    if (output_file == NULL) {
        printf("Error: Could not open train.bin\n");
        return 1;
    }

    fwrite(&total_tokens, sizeof(int), 1, output_file);
    fwrite(tokens, sizeof(uint16_t), total_tokens, output_file);

    fclose(output_file);

    printf("Saved train.bin successfully.\n");

    free(id_to_char);
    free(tokens);

    printf("Preprocessing complete.\n");

    return 0;
}
