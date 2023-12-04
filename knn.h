#pragma once
#include <vector>
#include<iostream>
#include <float.h>
#define type_vect unsigned char
class knn {
private:
    int k;
    std::vector<int>* neighbors;
    std::vector<type_vect>* trainingData;
    std::vector<double> distances;
    std::vector<type_vect>* testingData;
    std::vector<type_vect>* getImage(int i);
public:
    knn(int k);
    ~knn();

    void findKNearest(std::vector<type_vect>* data);
    void setTrainingData(std::vector<type_vect>* data);
    void setTestData(std::vector<type_vect>* data);
    void setK(int k);
    double minDistance();

    std::vector<int> predict();
    double euclideanDistance(std::vector<type_vect>* data, std::vector<type_vect>* input);

};

