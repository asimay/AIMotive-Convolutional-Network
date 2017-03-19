//
//  ConvolutionLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ConvolutionLayer.hpp"

Eigen::MatrixXf ConvolutionLayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = flattenReceptiveFields(input);
    valueInput.conservativeResize(valueInput.rows(), valueInput.cols() + 1);
    valueInput.col(valueInput.cols() - 1) = Eigen::VectorXf::Ones(valueInput.rows());
    valueOutput = valueInput * layerFilters;
    return valueOutput;
}

Eigen::MatrixXf ConvolutionLayer::flattenReceptiveFields(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf flattenMatrix = Eigen::MatrixXf::Zero(nextSize * nextSize, filterSize * filterSize * previousDepth);
    int newX;
    int newY;
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int shiftX = 0; shiftX < filterSize; shiftX++) {
                for (int shiftY = 0; shiftY < filterSize; shiftY++) {
                    for (int depth = 0; depth < previousDepth; depth++) {
                        newX = x * stride + shiftX - filterSize/2;
                        newY = y * stride + shiftY - filterSize/2;
                        if (newX < 0 || newX >= previousSize || newY < 0 || newY >= previousSize) break;
                        flattenMatrix(flatten2DCoordinates(x, y, nextSize), flatten3DCoordinates(shiftX, shiftY, depth, filterSize, previousDepth)) = input(flatten2DCoordinates(newX, newY, previousSize), depth);
                    }
                }
            }
        }
    }
    return flattenMatrix;
}

Eigen::MatrixXf ConvolutionLayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    deltaInput = delta;
    deltaOutput = deltaInput * layerFilters.transpose();
    deltaOutput.conservativeResize(deltaOutput.rows(), deltaOutput.cols() - 1);
    deltaOutput = reorderReceptiveFields(deltaOutput);
    return deltaOutput;
}

Eigen::MatrixXf ConvolutionLayer::reorderReceptiveFields(const Eigen::MatrixXf& delta) {
    Eigen::MatrixXf reorderMatrix = Eigen::MatrixXf::Zero(previousSize * previousSize, previousDepth);
    int newX;
    int newY;
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int shiftX = 0; shiftX < filterSize; shiftX++) {
                for (int shiftY = 0; shiftY < filterSize; shiftY++) {
                    for (int depth = 0; depth < previousDepth; depth++) {
                        newX = x * stride + shiftX - filterSize/2;
                        newY = y * stride + shiftY - filterSize/2;
                        if (newX < 0 || newX >= previousSize || newY < 0 || newY >= previousSize) break;
                        reorderMatrix(flatten2DCoordinates(newX, newY, previousSize), depth) += delta(flatten2DCoordinates(x, y, nextSize), flatten3DCoordinates(shiftX, shiftY, depth, filterSize, previousDepth));
                    }
                }
            }
        }
    }
    return reorderMatrix;
}

/*Eigen::MatrixXf ConvolutionLayer::flattenMatrix(Eigen::MatrixXf* inputMatrix, int inputSize, int inputDepth, int filterSize, int stride) {
    int flattenSize = (inputSize-1) / stride + 1;
    Eigen::MatrixXf flattenMatrix = Eigen::MatrixXf::Zero(flattenSize * flattenSize, filterSize * filterSize * inputDepth);
    
    for (int inputX = 0; inputX < inputSize; inputX += stride) {
        for (int inputY = 0; inputY < inputSize; inputY += stride) {
            flattenMatrix.row(flatten2DCoordinates(inputX/stride, inputY/stride, flattenSize)) = flattenReceptiveField(inputMatrix, inputSize, inputDepth,  inputX, inputY, filterSize);
        }
    }
    return flattenMatrix;
}

Eigen::VectorXf ConvolutionLayer::flattenReceptiveField(const Eigen::MatrixXf& input, int inputSize, int inputDepth, int inputX, int inputY, int filterSize) {
    int newX;
    int newY;
    Eigen::VectorXf flattenField = Eigen::VectorXf::Zero(filterSize * filterSize * inputDepth);
    
    for (int fieldX = 0; fieldX < filterSize; fieldX++) {
        for (int fieldY = 0; fieldY < filterSize; fieldY++) {
            newX = inputX + fieldX - (int)filterSize/2;
            newY = inputY + fieldY - (int)filterSize/2;
            for (int depth = 0; depth < inputDepth; depth++) {
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

Eigen::MatrixXf ConvolutionLayer::reorderMatrix(Eigen::MatrixXf *inputMatrix, int inputSize, int inputDepth, int filterSize, int stride) {
    Eigen::MatrixXf reorderedMatrix = Eigen::MatrixXf::Zero(inputSize * inputSize, inputDepth);
    for (int inputX = 0; inputX < inputSize; inputX += stride) {
        for (int inputY = 0; inputY < inputSize; inputY += stride) {
            reorderReceptiveField(inputMatrix, &reorderedMatrix, inputSize, inputDepth, inputX, inputY, filterSize, stride);
        }
    }
    return reorderedMatrix;
}

void ConvolutionLayer::reorderReceptiveField(Eigen::MatrixXf *inputMatrix, Eigen::MatrixXf *outputMatrix, int inputSize, int inputDepth, int inputX, int inputY, int filterSize, int stride) {
    int newX;
    int newY;
    
    for (int fieldX = 0; fieldX < filterSize; fieldX++) {
        for (int fieldY = 0; fieldY < filterSize; fieldY++) {
            newX = inputX + fieldX - (int)filterSize/2;
            newY = inputY + fieldY - (int)filterSize/2;
            for (int depth = 0; depth < inputDepth; depth++) {
                if (newX < 0 || newX >= inputSize || newY < 0 || newY >= inputSize) {
                    break;
                }
                (*outputMatrix)(flatten2DCoordinates(newX, newY, inputSize), depth) += (*inputMatrix)(flatten2DCoordinates(inputX/stride, inputY/stride, (inputSize - 1) / stride + 1), flatten3DCoordinates(fieldX, fieldY, depth, filterSize, inputDepth));
            }
        }
    }
}*/






