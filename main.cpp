#include "mnist_loader.h"
#include "neural_network.h"
#include "draw_image.cpp"
#include <chrono>
#include <thread>
using namespace this_thread;
using namespace chrono;
using namespace std;

// stdint.h has typedefs that specify exact-width integer types, useful for image processing & handling
// Citations regarding building a CNN and conceptual understanding:
//          https://cplusplus.com/reference/fstream/ for making the file handling functions/simplicity
//          https://www.kaggle.com/code/iamsouravbanerjee/convolutional-neural-network-for-dummies for
//          https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/
//          https://www.opencv-srf.com/p/introduction.html
//

void trainNetwork(Network& myNetwork, Processor& images, Processor& labels, int miniBatchSize, int numTests, float learningRate) {
    float myBatchCost(0.3f);
    float batchAccuracy;
    int imageIndex;
    for (int b = 0; b < numTests; b++)
    {
        myBatchCost = 0.0f;
        batchAccuracy = 0.0f;
        for (int m = 0; m < miniBatchSize; m++)
        {
            imageIndex = b * miniBatchSize + m;
            for (int i = 0; i < 784; i++)
            {
                myNetwork.setInputValue(images.getPixel(imageIndex * 784 + i), i);
            }

            myNetwork.forwardPropogation();
            myNetwork.calculateCost(labels.getLabel(imageIndex));
            myNetwork.backpropagateOutputLayer(learningRate, miniBatchSize);
            myNetwork.backpropagateHiddenLayers(learningRate, miniBatchSize);
            myBatchCost += myNetwork.getCost() / miniBatchSize;
            batchAccuracy += ((float)myNetwork.isCorrect() / (float)miniBatchSize);
        }
        myNetwork.applyPropagation();
        cout << "Batch : " << b + 1 << " / " << numTests << " Accuracy : " << batchAccuracy * 100 <<  "% Cost : " << myBatchCost << endl;
    }
}
void testNetwork(Network& myNetwork, Processor& testImages, Processor& testLabels) {
    int correct(0);
    float accuracy(0.0f);

    for (int m = 0; m < testImages.getNumElements(); m++)
    {
        for (int i = 0; i < 784; i++)
        {
            myNetwork.setInputValue(testImages.getPixel(m * 784 + i), i);
        }
        myNetwork.forwardPropogation();
        myNetwork.calculateCost(testLabels.getLabel(m));
        correct += myNetwork.isCorrect();
        drawImage(m, testImages, testLabels);
        cout << endl << "Machines Answer : " << myNetwork.getAnswer() << endl;
        cout << endl << "Accumulated Accuracy : " << 100 * ((float)(correct) / (m + 1)) << "% " << endl;
        sleep_until(system_clock::now() + seconds(2));
    }
}

int main() {
    //Training set images: 60,000 samples
    //Training set labels: 60,000 labels
    //Generated Test set images: 10,000 samples
    //Generated Test set labels: 10,000 labels
    //Total of: 150,000 data points linked

    cout << "Reading Mnist Database Files, Please Wait..." << endl << endl;

    // READ THIS!!!!!!
    //
    // Change the file paths to your specific path on your computer prior to running this!!
    //
    Processor images("C:\\CLionProjects\\DSA_Projects\\Neural_Network_IP\\training_images\\train-images-idx3-ubyte\\train-images-idx3-ubyte");
    Processor labels("C:\\CLionProjects\\DSA_Projects\\Neural_Network_IP\\training_labels\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte");

    Processor testImages("C:\\CLionProjects\\DSA_Projects\\Neural_Network_IP\\testing_images\\t10k-images-idx3-ubyte\\t10k-images-idx3-ubyte");
    Processor testLabels("C:\\CLionProjects\\DSA_Projects\\Neural_Network_IP\\testing_labels\\t10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte");

    int arg1, arg2;

    cout << "Input number of layers (more layers increases efficacy, but increases computation complexity. Recommended: 4): " << endl;
    cin >> arg1;
    cout << endl;
    cout << "Input number of neurons per hidden layer (more neurons can enhance the network's learning capacity, but create excessive computations. Recommended: 30-40): " << endl;
    cin >> arg2;
    cout << endl;

    Network myNetwork(arg1, 784, arg2, 10);
    // 1. Minimum of 3, but can be changed to higher number
    // 2. Keep this at 784 for the number of pixels in each of the training images
    // 3. Third number is the number of neurons in each hidden layer, this can be changed
    // 4. Keep this at 10 for 0 - 9 digits in database

    int numElements = images.getNumElements();
    int miniBatchSize;
    int numTests;
    float learningRate;

    cout << "Input number of training images per batch (Changes based on total batch size of database. Recommended: 10 - 30.): " << endl;
    cin >> miniBatchSize;
    cout << endl;
    cout << "Input number of training batches you want to run (Computationally expensive, but the more batches ran the more accurate the results. Recommended: 1500-3000): " << endl;
    cin >> numTests;
    cout << endl;
    cout << "Input learning rate (Alters the convergence speed. Recommended: 1): " << endl;
    cin >> learningRate;
    cout << endl;

    // Start training
    trainNetwork(myNetwork, images, labels, miniBatchSize, numTests, learningRate);
    cout << endl << "Done Training!" << endl << endl;

    // Test and see output
    testNetwork(myNetwork, testImages, testLabels);
    return 0;
}

