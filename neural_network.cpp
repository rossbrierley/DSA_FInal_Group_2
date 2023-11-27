#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include "neural_network.h"
using namespace std;

NeuronLayer::NeuronLayer(int neuronNum, int weightNum)
        : neuronCount(neuronNum),
          weightCount(weightNum),

        // Construct the weight matrix with dimensions (neuronCount x weightCount)
          weight(neuronCount, std::vector<float>(weightCount)),

        // Construct bias, activation, gradient vectors
          bias(neuronCount),
          a(neuronCount),
          z(neuronCount),
          activationGradientVector(neuronCount),
          weightGradientVector(neuronCount, vector<float>(weightCount)),
          biasGradientVector(neuronCount)
{
    resetGradients();
    randomize();
}

void NeuronLayer::randomize() {
    // Random biases between -0.5 and 0.5
    const float range = 0.5f;

    // Iterate over the neurons to randomize bias and weights
    for(int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
        // Generate a random bias
        bias[neuronIndex] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * range - (range / 2.0f);

        // Iterate over weights of the current neuron to randomize
        for(int weightIndex = 0; weightIndex < weightCount; weightIndex++) {
            // Generate a random weight
            weight[neuronIndex][weightIndex] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * range - (range / 2.0f);
        }
    }
}

// Perform neuron activation based on the previous layer's activations and weights
void NeuronLayer::activateNeuronsWithPrevLayer(NeuronLayer const *prevLayer) {
    for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
        z[neuronIndex] = 0.0f; // Initialize the weighted sum (z) for each neuron

        // Calculate the weighted sum z, by multiplying previous layer activations with weights
        for (int weightIndex = 0; weightIndex < weightCount; weightIndex++) {
            z[neuronIndex] += prevLayer->getActivation(weightIndex) * weight[neuronIndex][weightIndex];
        }
        // Add bias to the weighted sum
        z[neuronIndex] += bias[neuronIndex];

        // Apply the activation function to the weighted sum
        a[neuronIndex] = inverseTransformation(z[neuronIndex]);
    }
}

// Reset gradients for neurons, biases, and weights
void NeuronLayer::resetGradients() {
    for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
        // Reset activation gradient & bias
        activationGradientVector[neuronIndex] = 0.0f;
        biasGradientVector[neuronIndex] = 0.0f;

        // Reset gradients for each weight over all the neurons
        for (int weightIndex = 0; weightIndex < weightCount; weightIndex++) {
            weightGradientVector[neuronIndex][weightIndex] = 0.0f;
        }
    }
}

// Update weights and biases with learned values and reset gradients
void NeuronLayer::updateWeightsAndBiases() {
    for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++) {
        for (int weightIndex = 0; weightIndex < weightCount; weightIndex++) {
            // Reset weight gradients
            weight[neuronIndex][weightIndex] += weightGradientVector[neuronIndex][weightIndex];
            weightGradientVector[neuronIndex][weightIndex] = 0.0f;
        }

        // Add learned gradients
        bias[neuronIndex] += biasGradientVector[neuronIndex];
        biasGradientVector[neuronIndex] = 0.0f;
    }
}

// Update the weight gradient for a neuron and weight index
void NeuronLayer::updateWeightGradient(float value, int neuron, int weightIndex) {
    weightGradientVector[neuron][weightIndex] += value;
}

// Update the bias gradient for a neuron
void NeuronLayer::updateBiasGradient(float value, int neuron) {
    biasGradientVector[neuron] += value;
}


int NeuronLayer::getNumNeurons() const {
    return neuronCount;
}


// Set the activation gradient value for neurons
void NeuronLayer::setActivationGradient(float value, int neuron) {
    activationGradientVector[neuron] = value;
}

// Update the activation gradient for a neuron
void NeuronLayer::updateActivationGradient(float value, int neuron) {
    activationGradientVector[neuron] += value;
}

// Set the activation value for neurons
void NeuronLayer::setActivation(float value, int neuron) {
    a[neuron] = value;
}

// Get the activation value of a specific neuron
float NeuronLayer::getActivation(int neuron) const {
    float activation = a[neuron];
    return activation;
}

// Get the activation gradient for neurons
float NeuronLayer::getActivationGradient(int neuron) const {
    float activationGradient = activationGradientVector[neuron];
    return activationGradientVector[neuron];
}

// Get the weighted input for neurons
float NeuronLayer::getWeightedInput(int neuron) const {
    float input = z[neuron];
    return input;
}

// Get the weight value for a neuron
float NeuronLayer::getWeight(int neuronIndex, int weightIndex) const {
    float weightInd = weight[neuronIndex][weightIndex];
    return weightInd;
}

// Inverse transform
float NeuronLayer::inverseTransformation(float inputValue)
{
    float transformation = 1.0f / (1 + exp(-inputValue));
    return transformation;
}

Network::Network(int iNumLayers, int iNumInputs, int iNumHiddens, int iNumOutputs)
// Initialize basic vals
        : numLayers(iNumLayers),
          numInputs(iNumInputs),
          numHiddens(iNumHiddens),
          numOutputs(iNumOutputs)
{
    if (isInvalidParameters()) {
        cout << "Invalid Neuron Parameters" << endl;
        return;
    }
    initializeNetworkLayers();
}

bool Network::isInvalidParameters() const {
    return (numLayers < 3 || numInputs < 1 || numHiddens < 1 || numOutputs < 1);
}

void Network::initializeNetworkLayers() {

    // Initialize layers with specified neurons
    correctAns.resize(numOutputs);
    layerList.resize(numLayers);
    layerList[0] = new NeuronLayer(numInputs, 0);
    layerList[1] = new NeuronLayer(numHiddens, numInputs);
    for (int i = 1; i < (numLayers - 2); i++) {
        layerList[i + 1] = new NeuronLayer(numHiddens, numHiddens);
    }
    layerList[numLayers - 1] = new NeuronLayer(numOutputs, numHiddens);
}

// Destructor to clean up memory used by layers
Network::~Network() {
    int i = 0;
    while(i < numLayers) {
        delete layerList[i];
    }
}

// Set input values for the network
void Network::setInputValue(float inputValue, int index) {
    layerList[0]->setActivation(inputValue, index);
}

// Perform forward propagation
void Network::forwardPropogation() {
    for (int i = 1; i < numLayers; i++) {
        NeuronLayer const * previousLayer = layerList[i - 1];
        layerList[i]->activateNeuronsWithPrevLayer(previousLayer);
    }
}

// Calculate error using squared error
void Network::calculateCost(int realAnswer) {
    float totalSquaredError = 0.0f;
    int predictedIndex = 0;
    float maxActivation = layerList[numLayers - 1]->getActivation(0);

    // Find the index of the maximum activation
    for (int i = 1; i < numOutputs; ++i) {
        float currentActivation = layerList[numLayers - 1]->getActivation(i);
        if (currentActivation > maxActivation) {
            maxActivation = currentActivation;
            predictedIndex = i;
        }
    }

    // Set correctAns based on the real answer
    for (int i = 0; i < numOutputs; ++i) {
        if (i == realAnswer) {
            correctAns[i] = 1.0f;
        }
        else {
            correctAns[i] = 0.0f;
        }
        totalSquaredError += pow(correctAns[i] - layerList[numLayers - 1]->getActivation(i), 2);
    }
    // Calculate cost using total squared error
    cost = totalSquaredError * 0.5f;

    // Update correctness and predicted answer based on the maximum activation
    correct = (correctAns[predictedIndex] == 1.0f) ? 1 : 0;
    answer = predictedIndex;
}

// Functions to retrieve cost, correctness, and predicted answer
float Network::getCost() const {
    return cost;
}

int Network::isCorrect() const {
    return correct;
}

int Network::getAnswer() const {
    return answer;
}

void Network::backpropagateOutputLayer(float learningRate, int batchSize) {
    for (int j = 0; j < numOutputs; j++) {
        // Find the activation gradient for output neurons
        float activationGradient = (layerList[numLayers - 1]->getActivation(j) - correctAns[j]) * transformDerivative(layerList[numLayers - 1]->getWeightedInput(j));
        layerList[numLayers - 1]->setActivationGradient(activationGradient, j);

        // Update weights and biases for output layer
        for (int k = 0; k < numHiddens; k++) {
            float weightGradient = -(learningRate / (float)batchSize) * activationGradient * layerList[numLayers - 2]->getActivation(k);
            layerList[numLayers - 1]->updateWeightGradient(weightGradient, j, k);
        }
        float biasGradient = -(learningRate / (float)batchSize) * activationGradient;
        layerList[numLayers - 1]->updateBiasGradient(biasGradient, j);
    }
}

void Network::backpropagateHiddenLayers(float learningRate, int batchSize) {
    for (int i = (numLayers - 2); i > 0; i--) {
        for (int j = 0; j < layerList[i]->getNumNeurons(); j++) {
            // Reset the activation gradient for each hidden neuron
            layerList[i]->setActivationGradient(0.0f, j);

            // Compute activation gradient for hidden neurons
            for (int k = 0; k < layerList[i + 1]->getNumNeurons(); k++) {
                float activationGradient = layerList[i + 1]->getWeight(k, j) * layerList[i + 1]->getActivationGradient(k) * transformDerivative(layerList[i]->getWeightedInput(j));
                layerList[i]->updateActivationGradient(activationGradient, j);
            }

            // Update weights and biases for hidden layers
            for (int k = 0; k < layerList[i - 1]->getNumNeurons(); k++) {
                float weightGradient = -(learningRate / (float)batchSize) * layerList[i]->getActivationGradient(j) * layerList[i - 1]->getActivation(k);
                layerList[i]->updateWeightGradient(weightGradient, j, k);
            }
            float biasGradient = -(learningRate / (float)batchSize) * layerList[i - 1]->getActivationGradient(j);
            layerList[i]->updateBiasGradient(biasGradient, j);
        }
    }
}


// Apply learned weights and reset for the next iteration
void Network::applyPropagation() {
    for (int i = 1; i < numLayers; i++) {
        layerList[i]->updateWeightsAndBiases();
    }
}

// Activation derivative
float Network::transformDerivative(float inputValue) {
    float dsigmoid = transform(inputValue) * (1 - transform(inputValue));
    return dsigmoid;
}

// Activation function (sigmoid)
float Network::transform(float inputValue) {
    float sigmoid = 1.0f / (1 + exp(-inputValue));
    return sigmoid;
}