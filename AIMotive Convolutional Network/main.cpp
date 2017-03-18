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
#define LEARNING_RATE 0.1

int main(int argc, const char * argv[]) {
    
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, 10);
    Eigen::MatrixXf delta = Eigen::MatrixXf::Random(1, 10);
    ReLULayer layer = ReLULayer();
    std::cout << input << std::endl;
    std::cout << delta << std::endl;
    std::cout << layer.forwardPropagation(input) << std::endl;
    std::cout << layer.backwardPropagation(delta) << std::endl;
    
    return 0;
    
}



