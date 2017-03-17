//
//  FullyConnectedLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 13..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "FullyConnectedLayer.hpp"

void FullyConnectedLayer::forwardPropagation() {
    std::cout << layerName << " - Forward propagation" << std::endl;
    
    calculateLayerValue();
    
    if (nextLayer != NULL) {
        nextLayer->forwardPropagation();
    }
}

void FullyConnectedLayer::backwardPropagation() {
    std::cout << layerName << " - Backward propagation" << std::endl;
    
    calculateLayerDelta();
    
    if (previousLayer != NULL) {
        adjustWeights();
        previousLayer->backwardPropagation();
    }
}

void FullyConnectedLayer::calculateLayerValue() {
    if (previousLayer != NULL) {
        std::cout << layerName << " - Calculating layer value" << std::endl;
        layerValue = previousLayer->getLayerValue();
        layerValue.conservativeResize(layerValue.size() + 1);
        layerValue(layerValue.size() - 1) = 1.0;
        layerValue *= layerWeight;
        layerValue /= layerWeight.rows();
        
        for (int i = 0; i < layerValue.size(); i++) {
            if (layerValue(i) < 0.0) layerValue(i) = 0.0;
        }
        std::cout << "Layer value:" << std::endl << layerValue << std::endl;
    }
}

void FullyConnectedLayer::calculateLayerDelta() {
    if ((previousLayer != NULL)&&(nextLayer != NULL)) {
        std::cout << layerName << " - Calculating layer delta" << std::endl;
        layerDelta = nextLayer->getLayerDelta();
        layerDelta.conservativeResize(layerDelta.size() - 1);
        layerDelta *= layerWeight.transpose();
        layerDelta /= layerWeight.cols();
        
        for (int i = 0; i < layerDelta.size()-1; i++) {
            if (layerValue(i) < 0.0) layerDelta(i) = 0.0;
        }
    }
}

void FullyConnectedLayer::adjustWeights() {
    if (previousLayer != NULL) {
        std::cout << layerName << " - Adjusting weights" << std::endl;
        std::cout << "Before:" << std::endl << layerWeight << std::endl;
        layerWeight -= (layerDelta.transpose() * previousLayer->getLayerValue()) * learningRate;
        std::cout << "After:" << std::endl << layerWeight << std::endl;
        std::cout << "Difference:" << std::endl << (layerDelta.transpose() * previousLayer->getLayerValue()) * learningRate << std::endl;
    }
}
