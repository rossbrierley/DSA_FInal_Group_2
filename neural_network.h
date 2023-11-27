#pragma once
#include <vector>
using namespace std;

class NeuronLayer {
private:
    int neuronCount;
    int weightCount;
    vector<float> a;
    vector<float> z;
    vector<vector<float>> weight;
    vector<float> bias;
    vector<float> activationGradientVector;
    vector<vector<float>> weightGradientVector;
    vector<float> biasGradientVector;

public:
    NeuronLayer(int neuronNum, int weightNum);

    void randomize();
    void activateNeuronsWithPrevLayer(const NeuronLayer* prevLayer);
    void resetGradients();
    void updateWeightsAndBiases();
    void updateWeightGradient(float value, int neuron, int weightIndex);
    void updateBiasGradient(float value, int neuron);
    void setActivationGradient(float value, int neuron);
    void updateActivationGradient(float value, int neuron);
    void setActivation(float value, int neuron);

    int getNumNeurons() const;

    float getActivation(int neuron) const;
    float getActivationGradient(int neuron) const;
    float getWeightedInput(int neuron) const;
    float getWeight(int neuronIndex, int weightIndex) const;
    float inverseTransformation(float inputValue);
};

class Network {
private:
    vector<NeuronLayer*> layerList;
    vector<float> correctAns;
    int numInputs;
    int numHiddens;
    int numOutputs;
    int numLayers;
    int correct;
    int answer;
    float cost;

public:
    Network(int iNumLayers, int iNumInputs, int iNumHiddens, int iNumOutputs);
    ~Network();

    bool isInvalidParameters() const;

    void initializeNetworkLayers();
    void setInputValue(float inputValue, int index);
    void forwardPropogation();
    void calculateCost(int realAnswer);
    void backpropagateOutputLayer(float learningRate, int batchSize);
    void backpropagateHiddenLayers(float learningRate, int batchSize);
    void applyPropagation();

    int isCorrect() const;
    int getAnswer() const;

    float getCost() const;
    float transformDerivative(float inputValue);
    float transform(float inputValue);
};
