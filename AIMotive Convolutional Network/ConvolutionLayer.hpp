//
//  ConvolutionLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ConvolutionLayer_hpp
#define ConvolutionLayer_hpp

#include <stdio.h>
#include <Eigen>
#include <iostream>

#endif /* ConvolutionLayer_hpp */

using Eigen::MatrixXf;
using std::cout;
using std::endl;

class ConvolutionLayer {
private:
    unsigned int inputSize;
    unsigned int inputDepth;
    unsigned int filterSize;
    unsigned int filterNumber;
    unsigned int stride;
    
    float*** inputArray;
    float*** outputArray;
    
    MatrixXf inputMatrix;
    MatrixXf filterMatrix;
    MatrixXf outputMatrix;
public:
    ConvolutionLayer(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    ~ConvolutionLayer();
    void loadInputArray(float***);
    void flattenInputArray();
    void forwardConvolution();
    void reshapeOutputMatrix();
};
