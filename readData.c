#include "readData.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NUM_EXAMPLES 569
#define NUM_FEATURES 30 // 30 and one extra empty space
#define SIZE_TRAIN (int) (0.8 * NUM_EXAMPLES * sizeof(float))

float* createDataset(char* filename) {
    FILE* file = fopen(filename, "r");

    float dataset[NUM_EXAMPLES][NUM_FEATURES];

    if (file == NULL) {
        printf("Error opening file");
        return -1;
    }

    for (int i = 0; i < NUM_EXAMPLES; i++) {
        for (int j = 0; i < NUM_FEATURES; j++) {
            fscanf("%f", &dataset[i][j]);
        }   
    }
    fflush(NULL);
    fclose(file);
    return dataset;
}

float* createTrainset(float* dataset, int size_train) {
    float* trainset = malloc(size_train * sizeof(float));
    if (!trainset) {
        printf("Error declaring trainset");
        return -1;
    }
    memcpy(trainset, dataset, size_train * sizeof(float));
    return trainset;
}

float* createTestset(float* dataset, int size_test) {
    float* testset = malloc(size_test * sizeof(float));
    if (!testset) {
        printf("Error declaring testset");
        return -1;
    }
    memcpy(testset, dataset + (int) (0.8 * NUM_EXAMPLES * sizeof(float)), size_test * sizeof(float));
    return testset;
}