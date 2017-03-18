//
//  SoftmaxLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 18..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef SoftmaxLayer_hpp
#define SoftmaxLayer_hpp

#include <stdio.h>
#include <iostream>
#include <Eigen>

class SoftmaxLayer {
private:
    const std::string layerName;
    
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    
public:
    SoftmaxLayer() : layerName("") {}
    SoftmaxLayer(std::string layerName) : layerName(layerName) {}
    ~SoftmaxLayer() {}
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf&);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf&);
    
};

#endif /* SoftmaxLayer_hpp */
