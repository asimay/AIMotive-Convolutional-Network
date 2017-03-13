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
    
    Layer3D* previousLayer;
    Layer3D* nextLayer;
    
    Eigen::MatrixXf layerValue;
    Eigen::MatrixXf layerDelta;
    Eigen::MatrixXd layerIndex;
    
    int layerSize;
    int layerDepth;
    int poolingSize;
    
public:
    
    PoolingLayer(Layer3D* previousLayer, int poolingSize);
    ~PoolingLayer() {}
    
    Eigen::MatrixXf* getValue() { return &layerValue; }
    Eigen::MatrixXf* getDelta() { return &layerDelta; }
    
    int getSize() { return layerSize; }
    int getDepth() { return layerDepth; }
    
    void setNextLayer(Layer3D* nextLayer) {
        this->nextLayer = nextLayer;
    }
    
    void forwardPropagation();
    void backwardPropagation();
    
    static void findMaxValues(Eigen::MatrixXf* inputMatrix, Eigen::MatrixXf* outputMatrix, Eigen::MatrixXd* indexMatrix, int inputSize, int inputDepth, int poolingSize);
    static void findMaxInPool(Eigen::MatrixXf* inputMatrix, float* maxValue, int* maxIndex, int inputSize, int poolingSize, int poolX, int poolY, int poolingDepth);
};

#endif /* PoolingLayer_hpp */
