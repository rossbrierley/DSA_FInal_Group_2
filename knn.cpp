#include "knn.h"
#include <cmath>
#include <limits>
#include <map>

knn::knn(int k){
   this->k = k;
}
knn::~knn(){

}

std::vector<type_vect>* knn::getImage(int i){
    const int dataIndex = 16;
    if (this->trainingData->empty())
        return nullptr;

    int index = 784*i;
    std::vector<type_vect>* temp = new std::vector<type_vect>;

    for (int j = index+dataIndex; j < (index + 784 + dataIndex); j++){
        temp->push_back(this->trainingData->at(j));
    }
    return temp;
}
void knn::findKNearest(std::vector<type_vect>* data) {
    neighbors = new std::vector<int>;//Allocate memory for neighbors vector
    this->distances.clear();
    double min = FLT_MAX;//Set to infinity
    double previousMin = min;

    int index = 0;
    for (int i = 0; i < k; i++){

        if (i == 0){
            for(int j = 0; j < size(*this->trainingData)/784; j++){
               double distance = euclideanDistance(data,this->getImage(j));
               if (distance < min){
                   this->distances.push_back(distance);
                   min = distance;
                   index = j;
               }

            }
            neighbors->push_back(index);
            previousMin = min;
            min = FLT_MAX;
        }
        else{

            for(int j = 0; j < size(*this->trainingData)/784; j++){
                double distance = euclideanDistance(data,this->getImage(j));
                if (distance > previousMin && distance < min){
                    this->distances.push_back(distance);
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(index);
            previousMin = min;
            min = FLT_MAX;
        }
    }



}
void knn::setTrainingData(std::vector<type_vect>* data){
    this->trainingData = data;
}

void knn::setTestData(std::vector<type_vect>* data){
    this->testingData = data;
}


void knn::setK(int k){
this->k = k;
}
double knn::minDistance(){
    return this->distances[0];

}

std::vector<int> knn::predict(){
    //28:00

//    for (int i : *this->neighbors){
//
//        std::cout << "Prediction: " << i << std::endl;
//    }
    std::vector<int> temp;
    temp = *this->neighbors;
    delete this->neighbors;

    temp.push_back(this->distances[0]);
    return temp;

}

double knn::euclideanDistance(std::vector<type_vect>* data, std::vector<type_vect>* input){
    double distance = 0.0;

    if (size(*data) != size(*input)) {
        std::cout << "error\n";
        exit(1);
    }
    for (unsigned int i = 0; i < size(*data); i++){
        distance += pow(data->at(i) - input->at(i),2);

    }

    return sqrt(distance);
}

