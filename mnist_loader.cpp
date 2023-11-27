#pragma once
#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include "mnist_loader.h"
using namespace std;

// constructor reads binary files and processes header information
Processor::Processor(const char fileName[]) {

    char c;
    ifstream infile;
    infile.open(fileName, ios::binary | ios::in);
    if(!infile.is_open()) {
        cout << "Unable to open file. " << endl;
    }

    // get size of the file and create a vector to hold the file data
    fileSize = getFileSize(fileName);

    // resize fileVector to accommodate the file content
    fileVector.resize(fileSize);

    // read file content into the fileVector
    for (int i = 0; i < fileSize; i++) {
        infile.read(&c, 1);
        fileVector[i] = c;
    }
    infile.close();

    // extract to determine the file type
    fileTypeIndicator = getBigIntEndian(0);

    if ((fileTypeIndicator != 2051) && (fileTypeIndicator != 2049)) {
        cout << "Unsupported file type." << endl;
        return;
    }

    // get number of elements and determine the structure based on file type
    numElements = getBigIntEndian(4);

    if (fileTypeIndicator == 2051) {
        numRows = getBigIntEndian(8);
        numCols = getBigIntEndian(12);
        dataIndex = 16;
    }
    else {
        dataIndex = 8;
    }
}

// get total number of elements in the file
int Processor::getNumElements() const {
    return numElements;
}

float Processor::getPixel(int index) const {
    // calculate absolute index in the fileVector for the pixel value
    int absoluteIndex = index + dataIndex;

    if (absoluteIndex < 0 || absoluteIndex >= fileVector.size()) {
        return 0.0f;
    }
    // get pixel value directly and normalize it
    float normalizedPixel = static_cast<float>(fileVector[absoluteIndex]) / 255.0f;
    return normalizedPixel;
}


// get label value at a specific index
int Processor::getLabel(int nIndex) const {
    int label = nIndex + dataIndex;
    return fileVector[label];
}

// get the size of the file
ifstream::pos_type Processor::getFileSize(const char* filename) {
    ifstream in(filename, ifstream::binary);
    if (!in.is_open()) {
        return -1;
    }
    // seek to end of the file and get file size
    in.seekg(0, ifstream::end);
    ifstream::pos_type fileSize = in.tellg();

    in.close();
    return fileSize;
}

// get file vector int
int Processor::getBigIntEndian(int index) const {
    int intValue = 0;
    for (int i = 0; i < 4; i++)
    {
        // update integer value by shifting the existing value left by 8 bits and adding the new byte
        intValue = (intValue << 8) | fileVector[index + i];
    }
    return intValue;
}
