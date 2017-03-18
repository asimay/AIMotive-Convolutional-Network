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

class ReLULayer {
    
private:
    const std::string layerName;
    
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
public:
    
    ReLULayer() : layerName(""),  valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ReLULayer(std::string layerName) : layerName(layerName),  valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ~ReLULayer() {}
    
    Eigen::MatrixXf getValueInput() { return valueInput; }
    Eigen::MatrixXf getValueOutput() { return valueOutput; }
    Eigen::MatrixXf getDeltaInput() { return deltaInput; }
    Eigen::MatrixXf getDeltaOutput() { return deltaOutput; }
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf& delta);
    
};

#endif /* ReLULayer_hpp */
