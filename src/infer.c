#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "model/model.h"

// Temperature sampling
int sample_temperature(float *logits, int vocab_size, float temperature) {
    if (temperature == 0.0f) {
        // Greedy
        int best_idx = 0;
        float best_val = logits[0];
        for(int i = 1; i < vocab_size; i++) {
            if(logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }
    
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    float *probs = malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        float val = (logits[i] - max_logit) / temperature;
        if (val > 20.0f) val = 20.0f;
        if (val < -20.0f) val = -20.0f;
        probs[i] = expf(val);
        sum += probs[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= (sum < 1e-9f ? 1e-9f : sum);
    }
    
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int sampled_idx = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r < cdf) {
            sampled_idx = i;
            break;
        }
    }
    free(probs);
    return sampled_idx;
}

// Top-K struct for sorting
typedef struct {
    float prob;
    int index;
} ProbIndex;

int compare_prob(const void *a, const void *b) {
    ProbIndex *pa = (ProbIndex*)a;
    ProbIndex *pb = (ProbIndex*)b;
    if (pa->prob < pb->prob) return 1;
    if (pa->prob > pb->prob) return -1;
    return 0;
}

int sample_top_k(float *logits, int vocab_size, float temperature, int top_k) {
    if (top_k <= 0 || top_k >= vocab_size) {
        return sample_temperature(logits, vocab_size, temperature);
    }
    
    float max_logit = logits[0];
    for(int i = 1; i < vocab_size; i++) {
        if(logits[i] > max_logit) max_logit = logits[i];
    }
    
    ProbIndex *pi = malloc(vocab_size * sizeof(ProbIndex));
    for(int i = 0; i < vocab_size; i++){
        float val = (logits[i] - max_logit) / temperature;
        if (val > 20.0f) val = 20.0f;
        if (val < -20.0f) val = -20.0f;
        pi[i].prob = expf(val);
        pi[i].index = i;
    }
    
    qsort(pi, vocab_size, sizeof(ProbIndex), compare_prob);
    
    float sum = 0.0f;
    for(int i = 0; i < top_k; i++) {
        sum += pi[i].prob;
    }
    
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int sampled_idx = pi[top_k - 1].index;
    for(int i = 0; i < top_k; i++) {
        cdf += pi[i].prob / (sum < 1e-9f ? 1e-9f : sum);
        if(r < cdf) {
            sampled_idx = pi[i].index;
            break;
        }
    }
    free(pi);
    return sampled_idx;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    
    printf("Initializing Inference Engine...\n");
    Model model;
    model_init(&model);
    model_load(&model, "model.bin");
    
    // Load Vocabulary
    int act_vocab_size = 0;
    unsigned char id_to_char[256] = {0};
    int char_to_id[256] = {0};
    
    FILE *vf = fopen("data/processed/vocab.bin", "rb");
    if(!vf) {
        printf("Error: Could not load data/processed/vocab.bin!\n");
        return 1;
    }
    fread(&act_vocab_size, sizeof(int), 1, vf);
    fread(id_to_char, sizeof(unsigned char), act_vocab_size, vf);
    fclose(vf);
    
    // Build reverse mapping
    for(int i=0; i<act_vocab_size; i++) {
        char_to_id[id_to_char[i]] = i;
    }
    
    uint16_t context[SEQ_LEN];
    for(int i=0; i<SEQ_LEN; i++) {
        context[i] = char_to_id[' '];
    }
    
    printf("\nReady! Type a message and press Enter (or 'quit' to exit).\n");
    float temperature = 0.8f;
    int top_k = 10;
    int generate_length = 200;

    char user_input[256];

    while (1) {
        printf("\n\nUser: ");
        if (!fgets(user_input, sizeof(user_input), stdin)) break;
        
        int p_len = 0;
        while(user_input[p_len] != '\0' && user_input[p_len] != '\n') p_len++;
        user_input[p_len] = '\0';

        if (strcmp(user_input, "quit") == 0) break;

        // Feed prompt into context one character at a time (sliding window)
        for(int i=0; i < p_len; i++) {
            for (int j = 0; j < SEQ_LEN - 1; j++) {
                context[j] = context[j + 1];
            }
            context[SEQ_LEN - 1] = char_to_id[(unsigned char)user_input[i]];
        }
        
        printf("AI: ");
        
        for (int step = 0; step < generate_length; step++) {
            model_forward(&model, context);
            float *last_logits = &model.logits[(SEQ_LEN - 1) * VOCAB_SIZE];
            
            int next_token = sample_top_k(last_logits, VOCAB_SIZE, temperature, top_k);
            char c = id_to_char[next_token];
            
            printf("%c", c);
            fflush(stdout); 
            
            for (int i = 0; i < SEQ_LEN - 1; i++) {
                context[i] = context[i + 1];
            }
            context[SEQ_LEN - 1] = next_token;
        }
    }
    
    printf("\n\nExiting...\n");
    model_free(&model);
    return 0;
}
