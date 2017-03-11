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
#include "Layer.hpp"
#include "ImageLoader.hpp"
#include "ConvolutionLayer.hpp"

#define IMAGE_SIZE 2
#define NUMBER_OF_COLORS 3
#define NUMBER_OF_CLASSES 2
#define NUMBER_OF_IMAGES 100
#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"

int main(int argc, const char * argv[]) {
    
    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(16,3);
    //matrix << 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53, 61, 62, 63, 71, 72, 73, 81, 82, 83, 91, 92, 93, 101, 102, 103, 111, 112, 113, 121, 122, 123, 131, 132, 133, 141, 142, 143, 151, 152, 153, 161, 162, 163;
    std::cout << matrix << std::endl;
    Eigen::MatrixXf matrix2 = ConvolutionLayer::flattenMatrix(&matrix, 4, 3, 3);
    ConvolutionLayer::addBiasColumn(&matrix2);
    std::cout << matrix2 / 28 * Eigen::MatrixXf::Random(28, 1) << std::endl;
    
    return 0;
}



