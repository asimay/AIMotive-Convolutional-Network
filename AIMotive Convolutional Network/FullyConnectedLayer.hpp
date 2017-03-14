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
    
    const int layerSize; //without bias
    const float learningRate;
    
    
public:
    FullyConnectedLayer() : previousLayer(NULL), nextLayer(NULL), layerValue(Eigen::VectorXf()), layerWeight(Eigen::MatrixXf()), layerDelta(Eigen::VectorXf()), layerSize(0), learningRate(0.0) {}
    FullyConnectedLayer(FullyConnectedLayer* previousLayer, int layerSize, float learningRate) : previousLayer(previousLayer), nextLayer(NULL), layerValue(Eigen::VectorXf()), layerWeight(Eigen::MatrixXf()), layerDelta(Eigen::VectorXf()), layerSize(layerSize), learningRate(learningRate) {
        if (previousLayer != NULL) {
            layerWeight = Eigen::MatrixXf::Random(previousLayer->getLayerSize() + 1, layerSize);
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
    float getLearningRate() { return learningRate; }
    
    void forwardPropagation();
    void backwardPropagation();
    void adjustWeights();
    
};

#endif /* FullyConnectedLayer_hpp */
