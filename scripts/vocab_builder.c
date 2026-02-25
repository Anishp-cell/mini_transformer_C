#include <stdio.h>
#include <stdlib.h>

#define INPUT_PATH "../data/raw/input.txt"
#define OUTPUT_PATH "../data/processed/vocab.bin"

int main() {

    FILE *input_file = fopen(INPUT_PATH, "rb");
    if (input_file == NULL) {
        printf("Error: Could not open input file at %s\n", INPUT_PATH);
        return 1;
    }

    int char_exists[256] = {0};
    unsigned char id_to_char[256];
    int char_to_id[256];
    int c;

    /* ----------------------------
       Correct file reading loop
    ---------------------------- */
    while ((c = fgetc(input_file)) != EOF) {
        char_exists[(unsigned char)c] = 1;
    }

    fclose(input_file);

    /* ----------------------------
       Count vocab size (FIXED)
    ---------------------------- */
    int vocab_size = 0;

    for (int i = 0; i < 256; i++) {
        if (char_exists[i]) {
            vocab_size++;
        }
    }

    if (vocab_size == 0) {
        printf("Vocabulary is empty.\n");
        return 1;
    }

    printf("Vocabulary size = %d\n", vocab_size);

    /* ----------------------------
       Assign IDs
    ---------------------------- */
    int current_id = 0;

    for (int i = 0; i < 256; i++) {
        if (char_exists[i]) {
            id_to_char[current_id] = (unsigned char)i;
            char_to_id[i] = current_id;
            current_id++;
        }
    }

    /* ----------------------------
       Save vocab.bin
       Format:
       [int vocab_size]
       [unsigned char id_to_char[vocab_size]]
    ---------------------------- */
    FILE *output_file = fopen(OUTPUT_PATH, "wb");

    if (output_file == NULL) {
        printf("Error: Could not open output file at %s\n", OUTPUT_PATH);
        return 1;
    }

    fwrite(&vocab_size, sizeof(int), 1, output_file);
    fwrite(id_to_char, sizeof(unsigned char), vocab_size, output_file);

    fclose(output_file);

    printf("Vocabulary saved to %s\n\n", OUTPUT_PATH);

    /* ----------------------------
       Print detected characters
    ---------------------------- */
    printf("Detected characters:\n");

    for (int i = 0; i < vocab_size; i++) {

        unsigned char ch = id_to_char[i];

        if (ch == '\n') {
            printf("[%d] -> '\\n'\n", i);
        }
        else if (ch == '\r') {
            printf("[%d] -> '\\r'\n", i);
        }
        else if (ch == '\t') {
            printf("[%d] -> '\\t'\n", i);
        }
        else if (ch == ' ') {
            printf("[%d] -> 'space'\n", i);
        }
        else {
            printf("[%d] -> '%c'\n", i, ch);
        }
    }

    printf("\nDone.\n");

    return 0;
}
