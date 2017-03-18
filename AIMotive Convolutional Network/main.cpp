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
#define LAYER_SIZE 10

int main(int argc, const char * argv[]) {
    
    srand((unsigned int)time(NULL));
    
    FullyConnectedLayer layer("First layer", 5, 5, 0.01);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, 5);
    Eigen::MatrixXf delta = Eigen::MatrixXf::Random(1, 5)/10.0;
    layer.forwardPropagation(input);
    layer.backwardPropagation(delta);
    layer.adjustWeights();
    return 0;
    
}



