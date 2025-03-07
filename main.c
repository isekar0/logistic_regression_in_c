#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "readData.h"

#define NUM_EXAMPLES 569
#define NUM_FEATURES 30
#define SIZE_TRAIN (int) (NUM_EXAMPLES * 0.8)
#define SIZE_TEST (int) (NUM_EXAMPLES * 0.2)

typedef struct PARAMETERS {
    float w[NUM_FEATURES - 1];
    float b;
} PARAMETERS;

typedef struct HYPERPARAMETERS {
    float learning_rate;
    float normalization_factor;
    int epochs;
    int batch_size;
    float clip_threshold;
    int patience_counter;
} HYPERPARAMETERS;

typedef struct METRICS {
    float running_loss;
    int TP;
    int TN;
    int FP;
    int FN;
    float accuracy;
    float precision;
    float recall;
    float f1_score;
} METRICS;

PARAMETERS initWeights(void);
void overrideHyperparameters(HYPERPARAMETERS*);
float sigmoidFunction(float);
float lossFunction(int, float);
PARAMETERS trainLoop(float*, PARAMETERS*, HYPERPARAMETERS, METRICS*);
METRICS evaluate(float*, const PARAMETERS, const HYPERPARAMETERS, METRICS*);
void printMetrics(METRICS);

int main(void) {
    HYPERPARAMETERS hyperparameters = {1e-3, 0.2, 50, 64, 0.5, 10};
    METRICS metrics = {0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0};
    PARAMETERS weights = initWeights();

    float* dataset = createDataset("NormData.txt");
    float* trainset = createTrainset(dataset, SIZE_TRAIN);
    float* testset = createTestset(dataset, SIZE_TEST);

    overrideHyperparameters(&hyperparameters);
    
    weights = trainLoop(trainset, &weights, hyperparameters, &metrics);
    metrics = evaluate(testset, weights, hyperparameters, &metrics);
    printMetrics(metrics);
    
    return 0;
}

PARAMETERS initWeights(void) {
    PARAMETERS weights;
    for (int i = 0; i < NUM_FEATURES - 1; i++) {
        float random_weight = rand() / RAND_MAX;
        weights.w[i] = random_weight;
    }
    float random_weight = rand() / RAND_MAX;
    weights.b = random_weight;
    return weights; 
}
void overrideHyperparameters(HYPERPARAMETERS* hyperparameters) {
    char override = '\0';
    printf("Current learning rate is 0.001, Override it? [y/n]");
    scanf("%c", &override);
    if (override == 'y') {
        scanf("%f", &hyperparameters->learning_rate);
        if (hyperparameters->learning_rate <= 1e-7) {
            printf("Invalid input, reverting to default 0.001");
            hyperparameters->learning_rate = 1e-3;
        }
    }
    printf("Current normalization factor is 0.2, Override it? [y/n]");
    scanf("%c", &override);
    if (override == 'y') {
        scanf("%f", &hyperparameters->normalization_factor);
        if (hyperparameters->normalization_factor <= 0) {
            printf("Invalid input, reverting to default 0.2");
            hyperparameters->normalization_factor = 0.2;
        }
    }
    printf("Current #epochs is 50, Override it? [y/n]");
    scanf("%c", &override);
    if (override == 'y') {
        scanf("%d", &hyperparameters->epochs);
        if (hyperparameters->epochs <= 0) {
            printf("Invalid input, reverting to default 50");
            hyperparameters->epochs = 50;
        }
    }
    printf("Current batch size is 64, Override it? [y/n]");
    scanf("%c", &override);
    if (override == 'y') {
        scanf("%d", &hyperparameters->batch_size);
        if (hyperparameters->batch_size <= 0 || hyperparameters->batch_size > NUM_EXAMPLES) {
            printf("Invalid input, reverting to default 64");
            hyperparameters->batch_size = 64;
        }
    }
} 
float sigmoidFunction(float input) {
    return pow(1 + exp(-input), -1);
}
float lossFunction(int ground_truth, float prediction) {
    float epsilon = 1e-7; 
    return (float )((-ground_truth) * log(prediction) + (1 - ground_truth) * log(1 - prediction));
}

PARAMETERS trainLoop(float* trainset, PARAMETERS* weights, const HYPERPARAMETERS hyperparameters, METRICS* metrics) {
    int num_iterations = hyperparameters.epochs; // to make it a modiable value, since its const (cuz this is the only value to be modified)
    int batch_size = hyperparameters.batch_size;
    int predictions[SIZE_TRAIN];
    // float (*ptr_trainset)[NUM_EXAMPLES][NUM_FEATURES];
    // ptr_trainset = &trainset;
    int patience_counter = 0;
    float best_loss = 1e6;
    while (num_iterations-- || patience_counter > hyperparameters.patience_counter) {
        metrics->running_loss = 0; // make sure running loss is zero
        // first loop will go through the 569 examples with a step of 64 (batch size)
        for (int i = 0; i < SIZE_TRAIN; i += batch_size) {
            // at the last iteration of the for loop, there might be less than batch_size num. elements left, if so, adjust batch_size to be equal to the remainder
            batch_size = (i + batch_size >= SIZE_TRAIN)? (SIZE_TRAIN - i) : hyperparameters.batch_size;
            float* batch_results = (float* )malloc(batch_size * sizeof(float));
            // next loop will go through each example in the batch
            for (int j = 0; j < batch_size; j++) {
                // last loop goes through each feature in the example in the batch     
                float l2_norm = 0;
                for (int k = 1; k < NUM_FEATURES; k++) {
                    batch_results[j] += trainset[(i + j) * NUM_FEATURES + k] * weights->w[k - 1]; 
                    // L2 norm, sum of square of weights * norm factor/2
                    l2_norm += weights->w[k - 1] * weights->w[k - 1];
                }
                l2_norm = l2_norm * hyperparameters.normalization_factor / 2;
                // This is the forward prop part
                batch_results[j] += weights->b;
                batch_results[j] = sigmoidFunction(batch_results[j]);
                
                // Now compute the loss and then update the weights, remember the L2 norm
                metrics->running_loss += lossFunction(trainset[(i + j) * NUM_FEATURES + 0], batch_results[j]) + l2_norm;

                float diff_prediction = (batch_results[j] - trainset[(i + j) * NUM_FEATURES + 0]);
                for (int k = 1; k < NUM_FEATURES; k++) {
                    weights->w[k - 1] -= hyperparameters.learning_rate * (diff_prediction * trainset[(i + j) * NUM_FEATURES + k] 
                    + hyperparameters.normalization_factor * weights->w[k]);
                }
                weights->b -= hyperparameters.learning_rate * (diff_prediction);
            }
        }
        // If the loss decreased, then update best loss and keep going
        if (metrics->running_loss < best_loss) {
            best_loss = metrics->running_loss;
            metrics->running_loss = 0;
        }
        else {
            patience_counter++;
        }
        if (num_iterations % 10) {
            printf("Iteration #%d, Loss: %.2f", hyperparameters.epochs - num_iterations + 1, metrics->running_loss);
        }
    }
    return *weights;
}
METRICS evaluate(float* testset, const PARAMETERS weights, const HYPERPARAMETERS hyperparameters, METRICS* metrics) {
    // float (*ptr_testset)[SIZE_TEST][NUM_FEATURES];
    // ptr_testset = &testset;
    for (int i = 0; i < SIZE_TEST; i++) {
        float predictions[SIZE_TEST];
        for (int j = 1; j < NUM_FEATURES; j++) {
            predictions[i] += testset[i * NUM_FEATURES + j] * weights.w[j - 1] + weights.b;
        }
        predictions[i] = sigmoidFunction(predictions[i]);
        predictions[i] = (predictions[i] > hyperparameters.clip_threshold)? 1 : 0;
        if ((testset[i * NUM_FEATURES]) == 1) {
            if (predictions[i] == 1) {
                metrics->TP++; // GT is 1 & predicted 1 is true positive
            }
            else {
                metrics->FN++; // GT is 1 but predicted 0 is false negative
            }
        }
        else {
            if (predictions[i] == 1) {
                metrics->FP++; // GT is 0 but predicted 1 is false positive
            }
            else {
                metrics->TN++; // GT is 0 & predicted 0 is true negative
            }
        }
    }
    int TP = metrics->TP;
    int TN = metrics->TN;
    int FP = metrics->FP;
    int FN = metrics->FN;
    metrics->accuracy = (TP + TN) / (TP + FN + TN + FP);
    metrics->precision = TP / (TP + FP);
    metrics->recall = TP / (TP + FN);
    metrics->f1_score = (2 * metrics->precision * metrics->recall) / (metrics->precision + metrics->recall);
    return *metrics;
}

void printMetrics(METRICS metrics) {
    float accuracy = metrics.accuracy;
    float precision = metrics.precision;
    float recall = metrics.recall;
    float f1_score = metrics.f1_score;
    int TP = metrics.TP;
    int TN = metrics.TN;
    int FP = metrics.FP;
    int FN = metrics.FN;

    printf("\nMetrics:");
    printf("\nAccuracy: %.2f%%", accuracy * 100);
    printf("\nPrecision: %.2f%%", precision * 100);
    printf("\nRecall: %.2f%%", recall * 100);
    printf("\nF1-Score: %.2f%%", f1_score);
    printf("\nConfusion Matrix");
    printf("\n\tP'\tN'");
    printf("\nP\t%d\t%d", TP, FP);
    printf("\nN\t%d\t%d", FP, TN);
    printf("\n\n");
}
