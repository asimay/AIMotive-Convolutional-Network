//
//  FullyConnectedLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 13..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "FullyConnectedLayer.hpp"

Eigen::MatrixXf FullyConnectedLayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = input;
    valueInput.conservativeResize(valueInput.rows(), valueInput.cols() + 1);
    valueInput.col(valueInput.cols() - 1) = Eigen::VectorXf::Ones(valueInput.rows());
    valueOutput = valueInput * layerWeights;
    return valueOutput;
}

Eigen::MatrixXf FullyConnectedLayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    deltaInput = delta;
    deltaOutput = deltaInput * layerWeights.transpose();
    deltaOutput.conservativeResize(deltaOutput.rows(), deltaOutput.cols() - 1);
    return deltaOutput;
}


void FullyConnectedLayer::adjustWeights() {
    Eigen::MatrixXf delta = -valueInput.transpose() * deltaInput * learningRate;
    Eigen::MatrixXf deltaWeightProportion = delta;
    for (int row = 0; row < deltaWeightProportion.rows(); row++) {
        for (int col = 0; col < deltaWeightProportion.cols() ; col++) {
            deltaWeightProportion(row, col) /= layerWeights(row, col);
        }
    }
    deltaWeightProportion.cwiseAbs();
    //std::cout << layerName << ", min: " << deltaWeightProportion.minCoeff() << ", mean: " << deltaWeightProportion.mean() << ", max: " << deltaWeightProportion.maxCoeff() << std::endl;
    layerWeights *= 0.9999;
    layerWeights += delta;
}
