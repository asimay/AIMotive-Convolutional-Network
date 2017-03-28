//
//  ConvolutionLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ConvolutionLayer.hpp"

Eigen::MatrixXf ConvolutionLayer::forwardPropagation(const Eigen::MatrixXf& input) {
    valueInput = flattenReceptiveFields(input);
    valueInput.conservativeResize(valueInput.rows(), valueInput.cols() + 1);
    valueInput.col(valueInput.cols() - 1) = Eigen::VectorXf::Ones(valueInput.rows());
    valueOutput = valueInput * layerFilters;
    return valueOutput;
}

Eigen::MatrixXf ConvolutionLayer::flattenReceptiveFields(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf flattenMatrix = Eigen::MatrixXf::Zero(nextSize * nextSize, filterSize * filterSize * previousDepth);
    int newX;
    int newY;
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int shiftX = 0; shiftX < filterSize; shiftX++) {
                for (int shiftY = 0; shiftY < filterSize; shiftY++) {
                    for (int depth = 0; depth < previousDepth; depth++) {
                        newX = x * stride + shiftX - filterSize/2;
                        newY = y * stride + shiftY - filterSize/2;
                        if (newX < 0 || newX >= previousSize || newY < 0 || newY >= previousSize) break;
                        flattenMatrix(Layer3D::flatten2DCoordinates(x, y, nextSize), Layer3D::flatten3DCoordinates(shiftX, shiftY, depth, filterSize, previousDepth)) = input(Layer3D::flatten2DCoordinates(newX, newY, previousSize), depth);
                    }
                }
            }
        }
    }
    return flattenMatrix;
}

Eigen::MatrixXf ConvolutionLayer::backwardPropagation(const Eigen::MatrixXf& delta) {
    deltaInput = delta;
    deltaOutput = deltaInput * layerFilters.transpose();
    deltaOutput.conservativeResize(deltaOutput.rows(), deltaOutput.cols() - 1);
    deltaOutput = reorderReceptiveFields(deltaOutput);
    return deltaOutput;
}

Eigen::MatrixXf ConvolutionLayer::reorderReceptiveFields(const Eigen::MatrixXf& delta) {
    Eigen::MatrixXf reorderMatrix = Eigen::MatrixXf::Zero(previousSize * previousSize, previousDepth);
    int newX;
    int newY;
    for (int x = 0; x < nextSize; x++) {
        for (int y = 0; y < nextSize; y++) {
            for (int shiftX = 0; shiftX < filterSize; shiftX++) {
                for (int shiftY = 0; shiftY < filterSize; shiftY++) {
                    for (int depth = 0; depth < previousDepth; depth++) {
                        newX = x * stride + shiftX - filterSize/2;
                        newY = y * stride + shiftY - filterSize/2;
                        if (newX < 0 || newX >= previousSize || newY < 0 || newY >= previousSize) break;
                        reorderMatrix(Layer3D::flatten2DCoordinates(newX, newY, previousSize), depth) += delta(Layer3D::flatten2DCoordinates(x, y, nextSize), Layer3D::flatten3DCoordinates(shiftX, shiftY, depth, filterSize, previousDepth));
                    }
                }
            }
        }
    }
    return reorderMatrix;
}

void ConvolutionLayer::adjustFilters() {
    Eigen::MatrixXf delta = -valueInput.transpose() * deltaInput * learningRate;
    Eigen::MatrixXf deltaWeightProportion = delta;
    for (int row = 0; row < deltaWeightProportion.rows(); row++) {
        for (int col = 0; col < deltaWeightProportion.cols() ; col++) {
            deltaWeightProportion(row, col) /= layerFilters(row, col);
        }
    }
    deltaWeightProportion.cwiseAbs();
    //std::cout << layerName << ", min: " << deltaWeightProportion.minCoeff() << ", mean: " << deltaWeightProportion.mean() << ", max: " << deltaWeightProportion.maxCoeff() << std::endl;
    layerFilters *= 0.9999;
    layerFilters += delta;
}






