//
//  SignRecognition.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 28..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "SignRecognition.hpp"

void SignRecognition::performRecognition1() {
    
    ImageLoader imageLoader = ImageLoader(3, 5000, 52, 3);
    ConvolutionLayer layer1 = ConvolutionLayer("1st layer", 52, 3, 52, 3, 6, 1, LEARNING_RATE);
    PoolingLayer layer2 = PoolingLayer("2nd layer", 52, 26, 2, 6);
    ReLULayer layer3 = ReLULayer("3rd layer");
    ConvolutionLayer layer4 = ConvolutionLayer("4th layer", 26, 6, 26, 3, 12, 1, LEARNING_RATE);
    PoolingLayer layer5 = PoolingLayer("5th layer", 26, 13, 2, 12);
    ReLULayer layer6 = ReLULayer("6th layer");
    ConvolutionLayer layer7 = ConvolutionLayer("7th layer", 13, 12, 7, 5, 24, 2, LEARNING_RATE);
    ReLULayer layer8 ("8th layer");
    ConvolutionLayer layer9 = ConvolutionLayer("9th layer", 7, 24, 4, 5, 48, 2, LEARNING_RATE);
    PoolingLayer layer10 = PoolingLayer("10th layer", 4, 2, 2, 48);
    ReLULayer layer11 = ReLULayer("11th layer");
    FullyConnectedLayer layer12 = FullyConnectedLayer("12th layer", 48, 192, LEARNING_RATE);
    FullyConnectedLayer layer13 = FullyConnectedLayer("13th layer", 12, 48, LEARNING_RATE);
    SoftmaxLayer layer14 = SoftmaxLayer("14th layer");
    
    
    int success = 0;
    imageLoader.loadImages(FOLDER_PATH);
    for (int i = 0; i < 2000; i++) {
        Eigen::MatrixXf values = imageLoader.getImageMatrix(i%3, i);
        values = layer1.forwardPropagation(values);
        values = layer2.forwardPropagation(values);
        values = layer3.forwardPropagation(values);
        values = layer4.forwardPropagation(values);
        values = layer5.forwardPropagation(values);
        values = layer6.forwardPropagation(values);
        values = layer7.forwardPropagation(values);
        values = layer8.forwardPropagation(values);
        values = layer9.forwardPropagation(values);
        values = layer10.forwardPropagation(values);
        values = layer11.forwardPropagation(values);
        values.resize(1, 2 * 2 * 48);
        values = layer12.forwardPropagation(values);
        values = layer13.forwardPropagation(values);
        values = layer14.forwardPropagation(values);
        int max;
        int max2;
        values.maxCoeff(&max2, &max);
        std::cout << "Answer: " << i%3 << " Predicted: " << max <<std::endl;
        std::cout << values << std::endl;
        
        int row;
        int col;
        values.maxCoeff(&row, &col);
        if (col == i%3) success++;
        
        
        values(0, i%3) -= 1.0;
        
        values = layer14.backwardPropagation(values);
        values = layer13.backwardPropagation(values);
        values = layer12.backwardPropagation(values);
        values.resize(2 * 2, 48);
        values = layer11.backwardPropagation(values);
        values = layer10.backwardPropagation(values);
        values = layer9.backwardPropagation(values);
        values = layer8.backwardPropagation(values);
        values = layer7.backwardPropagation(values);
        values = layer6.backwardPropagation(values);
        values = layer5.backwardPropagation(values);
        values = layer4.backwardPropagation(values);
        values = layer3.backwardPropagation(values);
        values = layer2.backwardPropagation(values);
        values = layer1.backwardPropagation(values);
        layer1.adjustFilters();
        layer4.adjustFilters();
        layer7.adjustFilters();
        layer9.adjustFilters();
        layer12.adjustWeights();
        layer13.adjustWeights();
    }
    std::cout << success << std::endl;
}
