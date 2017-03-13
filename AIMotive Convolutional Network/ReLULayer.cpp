//
//  ReLULayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ReLULayer.hpp"

ReLULayer::ReLULayer(Layer3D* previousLayer) {
    this->previousLayer = previousLayer;
    previousLayer->setNextLayer(this);
    this->nextLayer = NULL;
    
    layerValue = Eigen::MatrixXf();
    layerDelta = Eigen::MatrixXf();
    
    layerSize = previousLayer->getSize();
    layerDepth = previousLayer->getDepth();
}

void ReLULayer::forwardPropagation() {
    layerValue = *previousLayer->getValue();
    for (int row = 0; row < layerSize * layerSize; row++) {
        for (int col= 0; col < layerDepth; col++) {
            if (layerValue(row, col) < 0.0 )
                layerValue(row, col) = 0.0;
        }
    }
}

void ReLULayer::backwardPropagation() {
    layerDelta = *nextLayer->getDelta();
    for (int row = 0; row < layerSize * layerSize; row++) {
        for (int col= 0; col < layerDepth; col++) {
            if (layerValue(row, col) < 0.0 )
                layerDelta(row, col) = 0.0;
        }
    }
}

