//
//  ConvolutionLayer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ConvolutionLayer.hpp"

ConvolutionLayer::ConvolutionLayer(unsigned int inputRows, unsigned int inputCols, unsigned int filterRows, unsigned int filterCols) {
    inputMatrix = MatrixXf(inputRows, inputCols);
    filterMatrix = MatrixXf::Random(filterRows, filterCols);
}

ConvolutionLayer::~ConvolutionLayer() {
    
}
