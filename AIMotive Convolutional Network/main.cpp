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
#define LEARNING_RATE 0.01
#define LAYER_SIZE 10

int main(int argc, const char * argv[]) {
    srand((unsigned int)time(NULL));
    
    FullyConnectedLayer layer1("First FC layer", LAYER_SIZE, LAYER_SIZE, LEARNING_RATE);
    ReLULayer layer2("First ReLU layer");
    FullyConnectedLayer layer3("Second FC layer", LAYER_SIZE, LAYER_SIZE, LEARNING_RATE);
    
    Eigen::MatrixXf input;
    Eigen::MatrixXf fc1;
    Eigen::MatrixXf relu1;
    Eigen::MatrixXf fc2;
    
    Eigen::MatrixXf delta;
    
    Eigen::MatrixXf fc2Delta;
    Eigen::MatrixXf relu1Delta;
    Eigen::MatrixXf fc1Delta;
    
    int success = 0;
    
    for (int i = 0; i < 3000; i++) {
        input = Eigen::MatrixXf::Random(1, LAYER_SIZE)/100;
        input(0, i%2) = 1.0;
        fc1 = layer1.forwardPropagation(input);
        relu1 = layer2.forwardPropagation(fc1);
        fc2 = layer3.forwardPropagation(relu1);
        
        std::cout << "Forward" << std::endl;
        std::cout << input << std::endl << std::endl;
        std::cout << fc1 << std::endl << std::endl;
        std::cout << relu1 << std::endl << std::endl;
        std::cout << fc2 << std::endl << std::endl;
        
        std::cout << "Answer: " << i%2 << std::endl;
        
        int placeholder;
        int answer;
        fc2.maxCoeff(&placeholder, &answer);
        if (i > 2800 && answer == i%2) success++;
        
        delta = Eigen::MatrixXf::Zero(1, LAYER_SIZE);
        delta(0, i%2) = 1.0;
        delta = fc2 - delta;
        std::cout << "Cost: " << delta * delta.transpose() << std::endl;
        
        fc2Delta = layer3.backwardPropagation(delta);
        relu1Delta = layer2.backwardPropagation(fc2Delta);
        fc1Delta = layer1.backwardPropagation(relu1Delta);
        
        std::cout << "Backward" << std::endl;
        std::cout << delta << std::endl << std::endl;
        std::cout << fc2Delta << std::endl << std::endl;
        std::cout << relu1Delta << std::endl << std::endl;
        std::cout << fc1Delta << std::endl << std::endl;
        
        layer1.adjustWeights();
        layer3.adjustWeights();
    }
    
    std::cout << "Success: " << success << std::endl;
    
    return 0;
    
}



