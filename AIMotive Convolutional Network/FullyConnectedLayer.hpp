//
//  FullyConnectedLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 13..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef FullyConnectedLayer_hpp
#define FullyConnectedLayer_hpp

#include <stdio.h>
#include <Eigen>
#include <iostream>

class FullyConnectedLayer {
private:
    FullyConnectedLayer* previousLayer;
    FullyConnectedLayer* nextLayer;
    
    Eigen::VectorXf layerValue;
    Eigen::MatrixXf layerWeight;
    Eigen::VectorXf layerDelta;
    
    int layerSize; //without bias
    
public:
    FullyConnectedLayer() : previousLayer(NULL), nextLayer(NULL), layerValue(Eigen::VectorXf()), layerWeight(Eigen::MatrixXf()), layerDelta(Eigen::VectorXf()), layerSize(0) {}
    FullyConnectedLayer(FullyConnectedLayer* previousLayer, int layerSize) : previousLayer(previousLayer), nextLayer(NULL), layerValue(Eigen::VectorXf()), layerWeight(Eigen::MatrixXf()), layerDelta(Eigen::VectorXf()), layerSize(layerSize) {
        if (previousLayer != NULL) {
            layerWeight = Eigen::MatrixXf::Random(previousLayer->getLayerSize(), layerSize);
            previousLayer->setNextLayer(this);
        }
    }
    ~FullyConnectedLayer() {}
    
    FullyConnectedLayer* getPreviousLayer() { return previousLayer; }
    void setPreviousLayer(FullyConnectedLayer* previousLayer) { this->previousLayer = previousLayer; }
    
    FullyConnectedLayer* getNextLayer() { return nextLayer; }
    void setNextLayer(FullyConnectedLayer* nextLayer) { this->nextLayer = nextLayer; }
    
    Eigen::VectorXf& getLayerValue() { return layerValue; }
    void setLayerValue(Eigen::VectorXf& layerValue) { this->layerValue = layerValue; }
    
    Eigen::MatrixXf& getLayerWeight() { return layerWeight; }
    void setLayerWeight(Eigen::MatrixXf& layerWeight) { this->layerWeight = layerWeight; }
    
    Eigen::VectorXf& getLayerDelta() { return layerDelta; }
    void setLayerDelta(Eigen::VectorXf& layerDelta) { this->layerDelta = layerDelta; }
    
    int getLayerSize() { return layerSize; }
    void setLayerSize(int layerSize) { this->layerSize = layerSize; }
    
    void forwardPropagation();
    void backwardPropagation();
    
    void calculateDeltaAndCost(Eigen::VectorXf& answer);
    
};

#endif /* FullyConnectedLayer_hpp */
