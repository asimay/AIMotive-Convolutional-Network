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

#endif /* ConvolutionLayer_hpp */

using Eigen::MatrixXf;

class ConvolutionLayer {
private:
    float*** inputArray;
    float*** outputArray;
    MatrixXf inputMatrix;
    MatrixXf filterMatrix;
    MatrixXf outputMatrix;
public:
    ConvolutionLayer(unsigned int, unsigned int, unsigned int, unsigned int);
    ~ConvolutionLayer();
    
};
