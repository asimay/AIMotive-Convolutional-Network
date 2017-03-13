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
#include "FullyConnectedLayer.hpp"

#define IMAGE_SIZE 2
#define NUMBER_OF_COLORS 3
#define NUMBER_OF_CLASSES 2
#define NUMBER_OF_IMAGES 100
#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"

int main(int argc, const char * argv[]) {
    
    Eigen::VectorXf vec1 = Eigen::VectorXf::Ones(5);
    vec1.conservativeResize(4);
    std::cout << vec1 << std::endl;
    
    return 0;
}



