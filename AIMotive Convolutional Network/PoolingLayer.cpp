//
//  PoolingLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "PoolingLayer.hpp"

PoolingLayer::PoolingLayer(Layer3D* previousLayer, int poolingSize) {
    this->previousLayer = previousLayer;
    previousLayer->setNextLayer(this);
    this->nextLayer = NULL;
    
    layerValue = Eigen::MatrixXf();
    layerDelta = Eigen::MatrixXf();
    
    layerSize = (previousLayer->getSize() - 1) / poolingSize + 1;
    layerDepth = previousLayer->getDepth();
    this->poolingSize = poolingSize;
}

void PoolingLayer::forwardPropagation() {
    layerValue = Eigen::MatrixXf::Zero(layerSize*layerSize, layerDepth);
    float maxValue;
    
}

void PoolingLayer::backwardPropagation() {}

void PoolingLayer::findMaxValues(Eigen::MatrixXf* inputMatrix, Eigen::MatrixXf* outputMatrix, Eigen::MatrixXf* indexMatrix, int inputSize, int inputDepth, int poolingSize) {
    float maxValue;
    int maxIndex;
    int outputSize = (inputSize - 1) / poolingSize + 1;
    for (int poolX = 0; poolX < outputSize; poolX++) {
        for (int poolY = 0; poolY < outputSize; poolY++) {
            for (int depth = 0; depth < inputDepth; depth++) {
                maxValue = -1;
                maxIndex = 0;
                findMaxInPool(inputMatrix, &maxValue, &maxIndex, inputSize, poolingSize, poolX, poolY, depth);
                (*outputMatrix)(flatten2DCoordinates(poolX, poolY, outputSize), depth) = maxValue;
                (*indexMatrix)(flatten2DCoordinates(poolX, poolY, outputSize), depth) = maxIndex;
            }
        }
    }
}

void PoolingLayer::findMaxInPool(Eigen::MatrixXf* inputMatrix, float* maxValue, int* maxIndex, int inputSize, int poolingSize, int poolX, int poolY, int poolingDepth) {
    int newX, newY;
    for (int shiftX = 0; shiftX < poolingSize; shiftX++) {
        for (int shiftY = 0; shiftY < poolingSize; shiftY++) {
            newX = flatten2DCoordinates(poolX, shiftX, poolingSize);
            newY = flatten2DCoordinates(poolY, shiftY, poolingSize);
            if (newX >= inputSize || newY >= inputSize) break;
            if ((*inputMatrix)(flatten2DCoordinates(newX, newY, inputSize), poolingDepth) > *maxValue) {
                *maxValue = (*inputMatrix)(flatten2DCoordinates(newX, newY, inputSize), poolingDepth);
                *maxIndex = flatten2DCoordinates(newX, newY, inputSize);
            }
        }
    }
}
