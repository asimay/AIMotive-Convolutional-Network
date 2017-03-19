//
//  PoolingLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "PoolingLayer.hpp"

Eigen::MatrixXf PoolingLayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = input;
    valueOutput = Eigen::MatrixXf::Zero(nextSize * nextSize, layerDepth);
    maxIndices = Eigen::MatrixXf::Zero(nextSize * nextSize, layerDepth);
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int depth = 0; depth < layerDepth; depth++) {
                valueOutput(flatten2DCoordinates(x, y, nextSize), depth) = valueInput(flatten2DCoordinates(x * poolingSize, y * poolingSize, previousSize), depth);
                maxIndices(flatten2DCoordinates(x, y, nextSize), depth) = flatten2DCoordinates(x * poolingSize, y * poolingSize, previousSize);
                for (int shiftX = 0; shiftX < poolingSize; shiftX++) {
                    for(int shiftY = 0; shiftY < poolingSize; shiftY++) {
                        if (valueInput(flatten2DCoordinates(x * poolingSize + shiftX, y * poolingSize + shiftY, previousSize), depth) > valueOutput(flatten2DCoordinates(x, y, nextSize), depth)) {
                            valueOutput(flatten2DCoordinates(x, y, nextSize), depth) = valueInput(flatten2DCoordinates(x * poolingSize + shiftX, y * poolingSize + shiftY, previousSize), depth);
                            maxIndices(flatten2DCoordinates(x, y, nextSize), depth) = flatten2DCoordinates(x * poolingSize + shiftX, y * poolingSize + shiftY, previousSize);
                        }
                    }
                }
            }
        }
    }
    return valueOutput;
}

/*void PoolingLayer::backwardPropagation() {
    int previousSize = previousLayer->getSize();
    int previousDepth = previousLayer->getDepth();
    layerDelta = Eigen::MatrixXf::Zero(previousSize*previousSize, previousDepth);
    for (int row = 0; row < layerSize*layerSize; row++) {
        for (int col = 0; col < layerDepth; col++) {
            layerDelta(layerIndex(row, col), col) = (*nextLayer->getDelta())(row, col);
        }
    }
}

void PoolingLayer::findMaxValues(Eigen::MatrixXf* inputMatrix, Eigen::MatrixXf* outputMatrix, Eigen::MatrixXd* indexMatrix, int inputSize, int inputDepth, int poolingSize) {
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
}*/

