//
//  SignRecognition.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 28..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "SignRecognition.hpp"

void SignRecognition::performSimpleRecognition() {
    ImageLoader imageLoader = ImageLoader(NUMBER_OF_CLASSES, NUMBER_OF_IMAGES, IMAGE_SIZE, NUMBER_OF_COLORS);
    ConvolutionLayer layer1 = ConvolutionLayer("1st layer", 52, 3, 52, 3, 48, 1, LEARNING_RATE);
    PoolingLayer layer2 = PoolingLayer("2nd layer", 52, 26, 2, 48);
    ReLULayer layer3 = ReLULayer("3rd layer");
    FullyConnectedLayer layer4 = FullyConnectedLayer("4th layer", 100, 26*26*48, LEARNING_RATE);
    ReLULayer layer5 = ReLULayer("5th layer");
    FullyConnectedLayer layer6 = FullyConnectedLayer("6th layer", 12, 100, LEARNING_RATE);
    SoftmaxLayer layer7 = SoftmaxLayer("7th layer");
    
    imageLoader.loadImages(FOLDER_PATH);
    int numberOfTrainingImages = (int)(NUMBER_OF_IMAGES * 0.8);
    float loss = 0.5;
    for (int epoch = 0; epoch < 10; epoch++) {
        for (int imageNumber = 0; imageNumber < numberOfTrainingImages; imageNumber++) {
            for (int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
                Eigen::MatrixXf values = imageLoader.getImageMatrix(classNumber, imageNumber);
                values = layer1.forwardPropagation(values);
                values = layer2.forwardPropagation(values);
                values = layer3.forwardPropagation(values);
                values.resize(1, 26*26*48);
                values = layer4.forwardPropagation(values);
                values = layer5.forwardPropagation(values);
                values = layer6.forwardPropagation(values);
                values = layer7.forwardPropagation(values);
                
                loss = 0.99 * loss + 0.01 * -1.0 * std::log(values(0, classNumber));
                std::cout << "Epoch: " << epoch+1 << ", Image: " << imageNumber+1 << ", Answer: " << classNumber + 1 << ", Prediction: " << values(0, classNumber) << ", Current loss: " << -1.0 * std::log(values(0, classNumber)) << ", Accumulated loss: " << loss << std::endl;
                
                values(0, classNumber) -= 1.0;
                values = layer7.backwardPropagation(values);
                values = layer6.backwardPropagation(values);
                values = layer5.backwardPropagation(values);
                values = layer4.backwardPropagation(values);
                values.resize(26*26, 48);
                values = layer3.backwardPropagation(values);
                values = layer2.backwardPropagation(values);
                values = layer1.backwardPropagation(values);
                
                layer1.adjustFilters();
                layer4.adjustWeights();
                layer6.adjustWeights();
                
                if (loss < 0.3) break;
                                                                                                                                                  
            }
        }
    }
    
    int maxRow;
    int maxCol;
    std::vector<int> matchNumbers = std::vector<int>(NUMBER_OF_CLASSES);
    for (int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
        for (int imageNumber = numberOfTrainingImages; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {
            Eigen::MatrixXf values = imageLoader.getImageMatrix(classNumber, imageNumber);
            values = layer1.forwardPropagation(values);
            values = layer2.forwardPropagation(values);
            values = layer3.forwardPropagation(values);
            values.resize(1, 26*26*48);
            values = layer4.forwardPropagation(values);
            values = layer5.forwardPropagation(values);
            values = layer6.forwardPropagation(values);
            values = layer7.forwardPropagation(values);
            
            values.maxCoeff(&maxRow, &maxCol);
            if (maxCol == classNumber) matchNumbers[classNumber]++;
        }
    }
    int sum = std::accumulate(matchNumbers.begin(), matchNumbers.end(), 0);
    std::cout << "Overall accuracy: " << (float)sum / (0.2 * NUMBER_OF_IMAGES) / NUMBER_OF_CLASSES * 100.0 << "%" << std::endl;
    for (int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
        std::cout << classNumber+1 << ": " << (float)matchNumbers[classNumber] / (0.2 * NUMBER_OF_IMAGES) * 100.0 << "%" << std::endl;
    }

}

void SignRecognition::performRecognition1() {
    
    ImageLoader imageLoader = ImageLoader(NUMBER_OF_CLASSES, NUMBER_OF_IMAGES, IMAGE_SIZE, NUMBER_OF_COLORS);
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
    ReLULayer layer13 = ReLULayer("13th layer");
    FullyConnectedLayer layer14 = FullyConnectedLayer("14th layer", 12, 48, LEARNING_RATE);
    SoftmaxLayer layer15 = SoftmaxLayer("15th layer");
    
    imageLoader.loadImages(FOLDER_PATH);
    int numberOfTrainingImages = (int)(NUMBER_OF_IMAGES * 0.8);
    float loss = 0.0;
    for (int epoch = 0; epoch < 1; epoch++) {
        for (int imageNumber = 0; imageNumber < numberOfTrainingImages; imageNumber++) {
            if (imageNumber % 10 == 9) loss = 0.0;
            for (int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
                Eigen::MatrixXf values = imageLoader.getImageMatrix(classNumber, imageNumber);
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
                values = layer15.forwardPropagation(values);
                
                loss += -1.0 * std::log(values(0, classNumber));
                
                values(0, classNumber) -= 1.0;
                
                values = layer15.backwardPropagation(values);
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
                layer14.adjustWeights();
            }
            if (imageNumber% 10 == 9) std::cout << "Loss: " << loss / NUMBER_OF_CLASSES / 10 << std::endl;
        }
    }
    
    std::vector<int> matchNumbers = std::vector<int>(12);
    for (int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
        for (int imageNumber = numberOfTrainingImages; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {
            Eigen::MatrixXf values = imageLoader.getImageMatrix(classNumber, imageNumber);
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
            values = layer15.forwardPropagation(values);
            
            int maxRow;
            int maxCol;
            values.maxCoeff(&maxRow, &maxCol);
            if (maxCol == classNumber) matchNumbers[classNumber]++;
        }
    }
    std::cout << "Sum: " << std::accumulate(matchNumbers.begin(), matchNumbers.end(), 0) << std::endl;
    for (int classNumber = 0; classNumber < 12; classNumber++) {
        std::cout << classNumber+1 << ": " << matchNumbers[classNumber] << std::endl;
    }
}
