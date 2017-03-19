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
#include "SoftmaxLayer.hpp"

#define IMAGE_SIZE 2
#define NUMBER_OF_COLORS 3
#define NUMBER_OF_CLASSES 2
#define NUMBER_OF_IMAGES 100
#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"
#define LEARNING_RATE 0.01
#define LAYER_SIZE 10

int main(int argc, const char * argv[]) {
    srand((unsigned int)time(NULL));
    
    ConvolutionLayer layer("", 5, 1, 5, 5, 10, 1);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(25, 1);
    Eigen::MatrixXf output = input;
    //input << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    for (int i = 0; i < 10; i++) {
        output = layer.forwardPropagation(output);
        std::cout << output << std::endl << std::endl;
    }
    
    return 0;
    
}



