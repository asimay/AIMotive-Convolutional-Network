//
//  ReLULayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 12..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ReLULayer.hpp"

Eigen::MatrixXf ReLULayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = input;
    valueOutput = valueInput;
    for (int i = 0; i < valueOutput.size(); i++ ) {
        if (valueInput(i) < 0.0) valueOutput(i) *= 0.01;
    }
    return valueOutput;
}

Eigen::MatrixXf ReLULayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    deltaInput = delta;
    deltaOutput = deltaInput;
    for (int i = 0; i < deltaOutput.size(); i++) {
        if (valueInput(i) < 0.0) deltaOutput(i) *= 0.01;
    }
    return deltaOutput;
}
