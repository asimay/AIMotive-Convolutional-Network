//
//  FullyConnectedLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 13..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "FullyConnectedLayer.hpp"

//DONE
void FullyConnectedLayer::forwardPropagation() {
    if (previousLayer != NULL) {
        layerValue = previousLayer->getLayerValue();
        layerValue.conservativeResize(layerValue.size() + 1);
        layerValue(layerValue.size() - 1) = 1.0;
        layerValue *= layerWeight;
        layerValue /= layerWeight.rows();
        
        for (int i = 0; i < layerValue.size(); i++) {
            if (layerValue(i) < 0.0) layerValue(i) = 0.0;
        }
    }
    
    if (nextLayer != NULL) {
        nextLayer->forwardPropagation();
    }
}

//DONE
void FullyConnectedLayer::backwardPropagation() {
    if (nextLayer != NULL) {
        layerDelta = nextLayer->getLayerDelta();
        layerDelta.conservativeResize(layerDelta.size() - 1);
        layerDelta *= layerWeight.transpose();
        layerDelta /= layerWeight.cols();
        
        for (int i = 0; i < layerDelta.size(); i++) {
            if (layerValue(i) < 0.0) layerDelta(i) = 0.0;
        }
    }
    
    if (previousLayer != NULL) {
        previousLayer->backwardPropagation();
    }
}
