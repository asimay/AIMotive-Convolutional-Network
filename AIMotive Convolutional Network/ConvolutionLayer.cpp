//
//  ConvolutionLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(unsigned int inputSize, unsigned int inputDepth, unsigned int filterSize, unsigned int filterNumber, unsigned int stride) {
    inputArray = new float**[inputSize];
    for (unsigned int x = 0; x < inputSize; x++) {
        inputArray[x] = new float*[inputSize];
        for (unsigned int y = 0; y < inputSize; y++) {
            inputArray[x][y] = new float[inputDepth];
        }
    }
    
    unsigned int outputSize = inputSize/stride;
    this->outputArray = new float**[outputSize];
    for (unsigned int x = 0; x < outputSize; x++) {
        this->outputArray[x] = new float*[outputSize];
        for (unsigned int y = 0; y < outputSize; y++) {
            outputArray[x][y] = new float[filterNumber];
        }
    }
    this->inputSize = inputSize;
    this->inputDepth = inputDepth;
    this->filterSize = filterSize;
    this->filterNumber = filterNumber;
    this->stride = stride;
    
    unsigned int inputMatrixSizes[2] = {(inputSize/stride)*(inputSize/stride), filterSize*filterSize*inputDepth};
    inputMatrix = MatrixXf(inputMatrixSizes[0], inputMatrixSizes[1]);
    filterMatrix = MatrixXf::Random(inputMatrixSizes[1], filterNumber);
}

ConvolutionLayer::~ConvolutionLayer() {
    for (unsigned int x = 0; x < inputSize; x++) {
        for (unsigned int y = 0; y < inputSize; y++) {
            delete [] inputArray[x][y];
        }
        delete [] inputArray[x];
    }
    delete [] inputArray;
}

void ConvolutionLayer::loadInputArray(float*** inputArray) {
    for (unsigned int x = 0; x < inputSize; x++) {
        for (unsigned int y = 0; y < inputSize; y++) {
            for (unsigned int depth = 0; depth < inputDepth; depth++) {
                this->inputArray[x][y][depth] = inputArray[x][y][depth];
            }
        }
    }
}

void ConvolutionLayer::flattenInputArray() {
    int x = 0;
    int y = 0;
    for (unsigned int inputX = 0; inputX < inputSize; inputX++) {
        for (unsigned int inputY = 0; inputY < inputSize; inputY++) {
            for (unsigned int filterX = 0; filterX < filterSize; filterX++) {
                for (unsigned int filterY = 0; filterY < filterSize; filterY++) {
                    for (unsigned int depth = 0; depth < inputDepth; depth++) {
                        x = inputX - filterSize/2 + filterX;
                        y = inputY - filterSize/2 + filterY;
                        if (x < 0 || x >= inputSize || y < 0 || y >= inputSize) {
                            inputMatrix(inputX*inputSize + inputY, filterX*filterSize*inputDepth + filterY*inputDepth + depth) = 0.0;
                            break;
                        }
                        inputMatrix(inputX*inputSize + inputY, filterX*filterSize*inputDepth + filterY*inputDepth + depth) = inputArray[x][y][depth];
                    }
                }
            }
        }
    }
}

void ConvolutionLayer::forwardConvolution() {
    outputMatrix = inputMatrix * filterMatrix;
}

void ConvolutionLayer::reshapeOutputMatrix() {
    unsigned int outputSize = inputSize/stride;
    for (unsigned int x = 0; x < outputSize; x++) {
        for (unsigned int y = 0; y < outputSize; y++) {
            for (unsigned int depth = 0; depth < filterNumber; depth ++) {
                outputArray[x][y][depth] = outputMatrix(x*outputSize + y, depth);
            }
        }
    }
}
