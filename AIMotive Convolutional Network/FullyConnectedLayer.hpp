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
    const std::string layerName;
    const int layerSize; //without bias
    const int previousLayerSize;
    const float learningRate;
    
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf layerWeights;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
    
public:
    FullyConnectedLayer() : layerName(""), layerSize(0), previousLayerSize(0), learningRate(0.0), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), layerWeights(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    FullyConnectedLayer(std::string layerName, int layerSize, int previousLayerSize, float learningRate) : layerName(layerName), layerSize(layerSize), previousLayerSize(previousLayerSize), learningRate(learningRate), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), layerWeights(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {
        layerWeights = Eigen::MatrixXf::Random(previousLayerSize + 1, layerSize) * sqrt(2.0/layerSize);
        std::cout << layerWeights.minCoeff() << " " << layerWeights.mean() << " " << layerWeights.maxCoeff() << std::endl;
    }
    ~FullyConnectedLayer() {}
    
    std::string getLayerName() { return layerName; }
    int getLayerSize() { return layerSize; }
    int getPreviousLayerSize() { return previousLayerSize; }
    float getLearningRate() { return learningRate; }
    
    Eigen::MatrixXf getValueInput() { return valueInput; }
    Eigen::MatrixXf getValueOutput() { return valueOutput; }
    Eigen::MatrixXf getLayerWeights() { return layerWeights; }
    Eigen::MatrixXf getDeltaInput() { return deltaInput; }
    Eigen::MatrixXf getDeltaOutput() { return deltaOutput; }
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf&);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf&);
    void adjustWeights();
    
};

#endif /* FullyConnectedLayer_hpp */
