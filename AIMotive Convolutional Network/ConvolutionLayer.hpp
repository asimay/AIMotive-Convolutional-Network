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

class ConvolutionLayer : public Layer3D {
    
private:
    const std::string layerName;
    
    const int previousSize;
    const int previousDepth;
    const int nextSize;
    const int filterSize;
    const int filterNumber;
    const int stride;
    
    Eigen::MatrixXf layerFilters;
    Eigen::MatrixXf valueInput;
    Eigen::MatrixXf valueOutput;
    Eigen::MatrixXf deltaInput;
    Eigen::MatrixXf deltaOutput;
    
public:
    
    ConvolutionLayer() : layerName(""), previousSize(0), previousDepth(0),  nextSize(0), filterSize(0), filterNumber(0), stride(0), layerFilters(Eigen::MatrixXf()), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {}
    ConvolutionLayer(std::string layerName, int previousSize, int previousDepth, int nextSize, int filterSize, int filterNumber, int stride) : layerName(layerName), previousSize(previousSize), previousDepth(previousDepth), nextSize(nextSize), filterSize(filterSize), filterNumber(filterNumber), stride(stride), layerFilters(Eigen::MatrixXf()), valueInput(Eigen::MatrixXf()), valueOutput(Eigen::MatrixXf()), deltaInput(Eigen::MatrixXf()), deltaOutput(Eigen::MatrixXf()) {
        layerFilters = Eigen::MatrixXf::Ones(filterSize * filterSize * previousDepth + 1, filterNumber) * sqrt(2.0/(filterSize * filterSize * previousDepth + 1)) / (filterSize * filterSize);
    }
    ~ConvolutionLayer() {}
    
    Eigen::MatrixXf forwardPropagation(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backwardPropagation(const Eigen::MatrixXf& delta);
    
    Eigen::MatrixXf flattenReceptiveFields(const Eigen::MatrixXf& input);
    
};



#endif /* ConvolutionLayer_hpp */
