//
//  ReLULayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ReLULayer.hpp"

ReLULayer::ReLULayer() {}

ReLULayer::~ReLULayer() {}

void ReLULayer::forwardPropagation() {
    outputMatrix = *(previousLayer->getOutput());
    for (unsigned int row = 0; row < imageSize * imageSize; row++) {
        for (unsigned int col= 0; col < imageDepth; col++) {
            if (outputMatrix(row, col) < 0.0 )
                outputMatrix(row, col) = 0.0;
        }
    }
}

void ReLULayer::backwardPropagation() {
    
}

