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
#include <random>

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
        
        layerWeights = Eigen::MatrixXf::Zero(previousLayerSize + 1, layerSize);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);
        for (int row = 0; row < layerWeights.rows(); row++) {
            for (int col = 0; col < layerWeights.cols(); col++) {
                layerWeights(row, col) = distribution(generator) * sqrt(2.0 / layerWeights.rows());
            }
        }
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
