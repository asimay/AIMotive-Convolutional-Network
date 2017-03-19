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

Eigen::MatrixXf PoolingLayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    deltaInput = delta;
    deltaOutput = Eigen::MatrixXf::Zero(previousSize * previousSize, layerDepth);
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int depth = 0; depth < layerDepth; depth++) {
                deltaOutput(maxIndices(flatten2DCoordinates(x, y, nextSize), depth), depth) = deltaInput(flatten2DCoordinates(x, y, nextSize), depth);
            }
        }
    }
    return deltaOutput;
}
