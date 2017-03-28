//
//  ConvolutionLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ConvolutionLayer_hpp
#define ConvolutionLayer_hpp

#include <Eigen>
#include "Layer3D.hpp"
#include <iostream>
#include <random>

class ConvolutionLayer {
    
private:
    const std::string layerName;
    
    const int previousSize;
    const int previousDepth;
    const int nextSize;
    const int filterSize;
    const int filterNumber;
    const int stride;
    const float learningRate;
    
    Eigen::MatrixXf layerFilters;
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
public:
    
    ConvolutionLayer() : layerName(""), previousSize(0), previousDepth(0),  nextSize(0), filterSize(0), filterNumber(0), stride(0), learningRate(0.0), layerFilters(Eigen::MatrixXf()), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ConvolutionLayer(std::string layerName, int previousSize, int previousDepth, int nextSize, int filterSize, int filterNumber, int stride, float learningRate) : layerName(layerName), previousSize(previousSize), previousDepth(previousDepth), nextSize(nextSize), filterSize(filterSize), filterNumber(filterNumber), stride(stride), learningRate(learningRate), layerFilters(Eigen::MatrixXf()), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {
        
        layerFilters = Eigen::MatrixXf::Zero(filterSize * filterSize * previousDepth + 1, filterNumber);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);
        for (int row = 0; row < layerFilters.rows(); row++) {
            for (int col = 0; col < layerFilters.cols(); col++) {
                layerFilters(row, col) = distribution(generator) * sqrt(2.0 / layerFilters.rows());
            }
        }
        
    }
    ~ConvolutionLayer() {}
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf& delta);
    
    Eigen::MatrixXf flattenReceptiveFields(const Eigen::MatrixXf& input);
    Eigen::MatrixXf reorderReceptiveFields(const Eigen::MatrixXf& delta);
    void adjustFilters();
    
};



#endif /* ConvolutionLayer_hpp */
