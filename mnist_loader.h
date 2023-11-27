#pragma once
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

class Processor {
public:
    Processor(const char fileName[]);

    int getNumElements() const;
    int getLabel(int nIndex) const;

    float getPixel(int index) const;

private:
    ifstream::pos_type getFileSize(const char* filename);

    int getBigIntEndian(int index) const;
    int fileSize;
    int numElements;
    int numRows;
    int numCols;
    int fileTypeIndicator;
    int dataIndex;

    std::vector<unsigned char> fileVector;
};
