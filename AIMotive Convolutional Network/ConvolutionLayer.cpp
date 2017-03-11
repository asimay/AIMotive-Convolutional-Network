//
//  ConvolutionLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(unsigned int filterSize, unsigned int filterNumber, unsigned int stride) {
    
    previousLayer = NULL;
    nextLayer = NULL;
    
    this->filterSize = filterSize;
    this->filterNumber = filterNumber;
    this->stride = stride;
    
    this->inputMatrix = Eigen::MatrixXf();
    this->filterMatrix = Eigen::MatrixXf();
    this->outputMatrix = Eigen::MatrixXf();
    this->deltaInputMatrix = Eigen::MatrixXf();
    this->deltaOutputMatrix = Eigen::MatrixXf();

}

ConvolutionLayer::~ConvolutionLayer() {
    
}

void ConvolutionLayer::forwardPropagation() {
    inputMatrix = flattenMatrix(previousLayer->getOutput(), previousLayer->getSize(), (unsigned int)previousLayer->getOutput()->cols(), this->filterSize, this->stride);
    addBiasColumn(&inputMatrix);
    
    outputMatrix = inputMatrix * filterMatrix;
    outputMatrix /= filterMatrix.rows();
}

void ConvolutionLayer::backwardPropagation() {
    
}

Eigen::MatrixXf ConvolutionLayer::flattenMatrix(Eigen::MatrixXf* inputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int filterSize, unsigned int stride) {
    unsigned int flattenSize = (inputSize-1) / stride + 1;
    Eigen::MatrixXf flattenMatrix = Eigen::MatrixXf::Zero(flattenSize * flattenSize, filterSize * filterSize * inputDepth);
    
    for (unsigned int inputX = 0; inputX < inputSize; inputX += stride) {
        for (unsigned int inputY = 0; inputY < inputSize; inputY += stride) {
            flattenMatrix.row(flatten2DCoordinates(inputX/stride, inputY/stride, flattenSize)) = flattenReceptiveField(inputMatrix, inputSize, inputDepth,  inputX, inputY, filterSize);
        }
    }
    return flattenMatrix;
}

Eigen::VectorXf ConvolutionLayer::flattenReceptiveField(Eigen::MatrixXf* inputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int inputX, unsigned int inputY, unsigned int filterSize) {
    int newX;
    int newY;
    Eigen::VectorXf flattenField = Eigen::VectorXf::Zero(filterSize * filterSize * inputDepth);
    
    for (unsigned int fieldX = 0; fieldX < filterSize; fieldX++) {
        for (unsigned int fieldY = 0; fieldY < filterSize; fieldY++) {
            newX = inputX + fieldX - (int)filterSize/2;
            newY = inputY + fieldY - (int)filterSize/2;
            for (unsigned int depth = 0; depth < inputDepth; depth++) {
                if (newX < 0 || newX >= inputSize || newY < 0 || newY >= inputSize) {
                    break;
                }
                flattenField(flatten3DCoordinates(fieldX, fieldY, depth, filterSize, inputDepth)) = (*inputMatrix)(flatten2DCoordinates(newX, newY, inputSize), depth);
            }
        }
    }
    return flattenField;
}

void ConvolutionLayer::addBiasColumn(Eigen::MatrixXf* inputMatrix) {
    inputMatrix->conservativeResize(inputMatrix->rows(), inputMatrix->cols() + 1);
    inputMatrix->col(inputMatrix->cols() - 1) = Eigen::VectorXf::Ones(inputMatrix->rows());
}

Eigen::MatrixXf ConvolutionLayer::reorderMatrix(Eigen::MatrixXf *inputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int filterSize, unsigned int stride) {
    Eigen::MatrixXf reorderedMatrix = Eigen::MatrixXf::Zero(inputSize * inputSize, inputDepth);
    for (unsigned int inputX = 0; inputX < inputSize; inputX += stride) {
        for (unsigned int inputY = 0; inputY < inputSize; inputY += stride) {
            reorderReceptiveField(inputMatrix, &reorderedMatrix, inputSize, inputDepth, inputX, inputY, filterSize, stride);
        }
    }
    return reorderedMatrix;
}

void ConvolutionLayer::reorderReceptiveField(Eigen::MatrixXf *inputMatrix, Eigen::MatrixXf *outputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int inputX, unsigned int inputY, unsigned int filterSize, unsigned int stride) {
    int newX;
    int newY;
    
    for (unsigned int fieldX = 0; fieldX < filterSize; fieldX++) {
        for (unsigned int fieldY = 0; fieldY < filterSize; fieldY++) {
            newX = inputX + fieldX - (int)filterSize/2;
            newY = inputY + fieldY - (int)filterSize/2;
            for (unsigned int depth = 0; depth < inputDepth; depth++) {
                if (newX < 0 || newX >= inputSize || newY < 0 || newY >= inputSize) {
                    break;
                }
                (*outputMatrix)(flatten2DCoordinates(newX, newY, inputSize), depth) += (*inputMatrix)(flatten2DCoordinates(inputX/stride, inputY/stride, (inputSize - 1) / stride + 1), flatten3DCoordinates(fieldX, fieldY, depth, filterSize, inputDepth));
            }
        }
    }
}





