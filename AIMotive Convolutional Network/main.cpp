//
//  main.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 14..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include <iostream>
#include <Eigen>

#include "Signrecognition.hpp"
#include "Layer3D.hpp"
#include "ImageLoader.hpp"
#include "ConvolutionLayer.hpp"

#define IMAGE_SIZE 2
#define NUMBER_OF_COLORS 3
#define NUMBER_OF_CLASSES 2
#define NUMBER_OF_IMAGES 100
#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"

int main(int argc, const char * argv[]) {
    
    Eigen::MatrixXf mat1 = Eigen::MatrixXf::Zero(16, 1);
    Eigen::MatrixXf mat2 = ConvolutionLayer::flattenMatrix(&mat1, 4, 1, 3, 1);
    Eigen::MatrixXf mat3 = ConvolutionLayer::reorderMatrix(&mat2, 4, 1, 3, 1);
    ConvolutionLayer::addBiasColumn(&mat2);
    Eigen::MatrixXf mat4 = ConvolutionLayer::reorderMatrix(&mat2, 4, 1, 3, 1);
    
    std::cout << mat1 << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << mat3 << std::endl << std::endl;
    std::cout << mat4 << std::endl;
    
    
    return 0;
}



