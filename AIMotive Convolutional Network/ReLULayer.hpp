//
//  ReLULayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ReLULayer_hpp
#define ReLULayer_hpp

#include <stdio.h>
#include "Layer3D.hpp"
#include <Eigen>

class ReLULayer : public Layer3D {
private:
    
    Layer3D* previousLayer;
    Layer3D* nextLayer;
    
    unsigned int imageSize;
    unsigned int imageDepth;
    
    Eigen::MatrixXf outputMatrix;
    Eigen::MatrixXf deltaOutputMatrix;
    
public:
    
    ReLULayer();
    ~ReLULayer();
    Eigen::MatrixXf* getOutput() { return &outputMatrix; }
    unsigned int getSize() { return imageSize; }
    unsigned int getDepth() { return imageDepth; }
    
    void setPreviousLayer(Layer3D* previousLayer) { this->previousLayer = previousLayer; }
    void setNextLayer(Layer3D* nextLayer) { this->nextLayer = nextLayer; }
    
    void forwardPropagation();
    void backwardPropagation();
    
};

#endif /* ReLULayer_hpp */
