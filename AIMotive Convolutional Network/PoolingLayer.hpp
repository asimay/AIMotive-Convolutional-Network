//
//  PoolingLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef PoolingLayer_hpp
#define PoolingLayer_hpp

#include <stdio.h>
#include "Layer3D.hpp"
#include <iostream>

class PoolingLayer : public Layer3D {

private:
    const std::string layerName;
    
    const int previousSize;
    const int nextSize;
    const int poolingSize;
    const int layerDepth;
    
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf maxIndices;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
public:
    PoolingLayer() : layerName(""), previousSize(0), nextSize(0), poolingSize(0), layerDepth(0), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), maxIndices(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    PoolingLayer(std::string layerName, int previousSize, int nextSize, int poolingSize, int layerDepth) : layerName(layerName), previousSize(previousSize), nextSize(nextSize), poolingSize(poolingSize), layerDepth(layerDepth), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), maxIndices(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ~PoolingLayer() {}
    
    Eigen::MatrixXf getValueInput() { return valueInput; }
    Eigen::MatrixXf getValueOutput() { return valueOutput; }
    Eigen::MatrixXf getMaxIndices() { return maxIndices; }
    Eigen::MatrixXf getDeltaInput() { return deltaInput; }
    Eigen::MatrixXf getDeltaOutput() { return deltaOutput; }
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf& input);
    void backwardPropagation();
};

#endif /* PoolingLayer_hpp */
