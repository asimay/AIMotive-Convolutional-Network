//
//  SoftmaxLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 18..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "SoftmaxLayer.hpp"

Eigen::MatrixXf SoftmaxLayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = input;
    valueOutput = valueInput;
    for (int i = 0; i < valueOutput.size(); i++) {
        valueOutput(i) = exp(valueOutput(i));
    }
    valueOutput /= valueOutput.sum();
    return valueOutput;
}

Eigen::MatrixXf SoftmaxLayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    return delta;
}
