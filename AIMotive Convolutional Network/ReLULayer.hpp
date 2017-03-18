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
    
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
public:
    
    ReLULayer() : valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ~ReLULayer() {}
    
    Eigen::MatrixXf getValueInput() { return valueInput; }
    Eigen::MatrixXf getValueOutput() { return valueOutput; }
    Eigen::MatrixXf getDeltaInput() { return deltaInput; }
    Eigen::MatrixXf getDeltaOutput() { return deltaOutput; }
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf&);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf&);
    
};

#endif /* ReLULayer_hpp */
