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
    
    Eigen::MatrixXf layerValue;
    Eigen::MatrixXf layerDelta;
    
    unsigned int layerSize;
    unsigned int layerDepth;
    
public:
    
    ReLULayer(Layer3D* previousLayer);
    ~ReLULayer();
    
    Eigen::MatrixXf* getValue() { return &layerValue; }
    Eigen::MatrixXf* getDelta() { return &layerDelta; }
    
    unsigned int getSize() { return layerSize; }
    unsigned int getDepth() { return layerDepth; }
    
    void setNextLayer(Layer3D* nextLayer) {
        this->nextLayer = nextLayer;
    }
    
    void forwardPropagation();
    void backwardPropagation();
    
};

#endif /* ReLULayer_hpp */
