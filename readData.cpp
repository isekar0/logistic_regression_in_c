#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <cmath>
using namespace std;

typedef pair<char, vector<double>> ROW; 
typedef vector<ROW> DATA;
typedef vector<pair<int, vector<double>>> ENCODED_DATA;

tuple<DATA, int> read_file_into_vector(const string filename) {
    ROW row;
    DATA dataset;
    vector<string> items_as_string;
    ifstream input_file;

    input_file.open(filename);
    if (input_file.fail()) {
        cout << "Error: opening file" << endl;
        exit(EXIT_FAILURE);
    }

    string one_string;
    int counter = 0, number_examples = 0;
    while (getline(input_file, one_string, ',')) {
        items_as_string.push_back(one_string);
        switch (counter) {
            case 0: break; 
            case 1: row.first = one_string[0]; 
            // cout << "row.first: " << row.first << endl; 
            break;
            default: row.second.push_back(strtod(one_string.c_str(), NULL));
        }
        // cout << "counter: " << counter << ", row.second.size(): " << row.second.size() << ", one_string: " << one_string << endl;
        if (row.second.size() == 29) {
            dataset.push_back(row);
            row.second.clear();
            items_as_string.clear();
            counter = 0;
            number_examples += 1;
        }
        else {counter += 1;};
    }
    input_file.close();

    return make_tuple(dataset, number_examples);
}
vector<double> average_data(const DATA dataset) {
    int number_features = dataset[0].second.size(), number_examples = dataset.size();
    vector<double> averages(number_features, 0);
    vector<double> rolling_sums(number_features, 0);
    
    for (int m = 0; m < number_examples; m++) {
        for (int n = 0; n < number_features; n++) {
            rolling_sums[n] += dataset[m].second[n];
        }
    }
    for (int i = 0; i < rolling_sums.size(); i++) {
        double mean_feature = rolling_sums[i] / (double) number_examples; 
        averages[i] = mean_feature;
    }
    return averages;
}
vector<double> standard_deviation_data(const DATA dataset, const vector<double> averages) {
    int number_features = dataset[0].second.size(), number_examples = dataset.size();
    vector<double> standard_deviation_features;
    vector<double> rolling_sums(number_features, 0);
    
    for (int m = 0; m < number_examples; m++) {
        for (int n = 0; n < number_features; n++) {
            rolling_sums[n] += pow((dataset[m].second[n] - averages[n]), 2);
        }
    }
    for (int i = 0; i < rolling_sums.size(); i++) {
        double mean_feature = rolling_sums[i] / (double) number_examples; 
        if (mean_feature < 0) {
            mean_feature = -mean_feature;
        }
        double square_mean_feature = pow(mean_feature, (double) 0.5);
        standard_deviation_features.push_back(square_mean_feature);
    }
    return standard_deviation_features;
}
DATA normalize_data(const DATA dataset, const vector<double> averages, const vector<double> standard_deviations) {
    int number_features = dataset[0].second.size(), number_examples = dataset.size();
    DATA normalized_dataset = dataset;
    for (int m = 0; m < number_examples; m++) {
        for (int n = 0; n < number_features; n++) {
            normalized_dataset[m].second[n] = (dataset[m].second[n] - averages[n]) / standard_deviations[n]; 
        }
    }
    return normalized_dataset;
}
ENCODED_DATA  encode_target_data(const DATA dataset) {
    int number_features = dataset[0].second.size(), number_examples = dataset.size();
    ENCODED_DATA encoded_data;
    for (int m = 0; m < number_examples; m++) {
        if (dataset[m].first == 'M') {
            pair _pair = make_pair(1, vector<double> (number_features, 0));
            encoded_data.push_back(_pair);
        }
        else {
            pair _pair = make_pair(0, vector<double> (number_features, 0));
            encoded_data.push_back(_pair);
        }
        for (int n = 0; n < number_features; n++) {
            encoded_data[m].second[n] = dataset[m].second[n];
        }
    }
    return encoded_data;
}

template <typename D>
void printDataset(const D dataset, const char *output_file) {
    FILE* original_stdout = fdopen(dup(fileno(stdout)), "w");
    freopen(output_file, "w", stdout);
    for (int i = 0; i < dataset.size(); i++) {
        cout << dataset[i].first << endl;
        for (int j = 0; j < dataset[i].second.size(); j++) {
            cout <<(float) dataset[i].second[j] << endl;
        }
        // cout << endl;
    }
    // cout << endl;
    fflush(stdout);
    dup2(fileno(original_stdout), fileno(stdout));
    fclose(original_stdout);
}

int main() {
    string filename = "wdbc.data";
    DATA dataset;
    int number_examples;
    tie(dataset, number_examples) = read_file_into_vector(filename);
    printDataset(dataset, "readDataOutputTest.txt");
    // cout << "Passed printdataset" << endl;
    // cout << "m: " << number_examples << endl;

    
    vector<double> averages = average_data(dataset);
    // cout << "Passed average_data" << endl;
    vector<double> standard_deviations = standard_deviation_data(dataset, averages);
    // cout << "Passed standard_deviation_data" << endl;
    DATA dataset_norm = normalize_data(dataset, averages, standard_deviations);
    // cout << "Passed normalize_data" << endl; 
    ENCODED_DATA dataset_encoded_norm = encode_target_data(dataset_norm);
    // cout << "Passed encode_target_data" << endl;  
    printDataset(dataset_encoded_norm, "NormData.txt");
    // cout << "Passed printDataset" << endl; 
    return 0;
}