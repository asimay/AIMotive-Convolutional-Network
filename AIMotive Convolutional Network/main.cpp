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
#include "ReLULayer.hpp"
#include "PoolingLayer.hpp"

#define IMAGE_SIZE 2
#define NUMBER_OF_COLORS 3
#define NUMBER_OF_CLASSES 2
#define NUMBER_OF_IMAGES 100
#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"

int main(int argc, const char * argv[]) {
    srand(time(NULL));
    
    Eigen::MatrixXf mat1 = Eigen::MatrixXf::Random(16, 1);
    mat1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    Eigen::MatrixXf mat2 = Eigen::MatrixXf::Zero(4, 1);
    Eigen::MatrixXf mat3 = Eigen::MatrixXf::Zero(4, 1);
    std::cout << mat1 << std::endl << std::endl;
    PoolingLayer::findMaxValues(&mat1, &mat2, &mat3, 4, 1, 1);
    std::cout << mat2 << std::endl << std::endl;
    std::cout << mat3 << std::endl << std::endl;
    
    return 0;
}



