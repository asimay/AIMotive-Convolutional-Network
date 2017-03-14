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
    
    FullyConnectedLayer firstLayer = FullyConnectedLayer("First layer", NULL, 5, LEARNING_RATE);
    FullyConnectedLayer secondLayer = FullyConnectedLayer("Second layer", &firstLayer, 5, LEARNING_RATE);
    FullyConnectedLayer thirdLayer = FullyConnectedLayer("Third layer", &secondLayer, 5, LEARNING_RATE);
    
    Eigen::RowVectorXf layer1 = Eigen::RowVectorXf::Random(5);
    Eigen::RowVectorXf diff = Eigen::RowVectorXf::Random(6);
    firstLayer.setLayerValue(layer1);
    thirdLayer.setLayerDelta(diff);
    
    firstLayer.forwardPropagation();
    thirdLayer.backwardPropagation();
    
    return 0;
    
}



