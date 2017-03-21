//
//  SignRecognition.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 28..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef SignRecognition_hpp
#define SignRecognition_hpp

#include "ImageLoader.hpp"
#include "ConvolutionLayer.hpp"
#include "ReLULayer.hpp"
#include "PoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"
#include "SoftmaxLayer.hpp"

#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"
#define LEARNING_RATE 0.001

class SignRecognition {
private:
    
public:
    
    static void performRecognition1();
    
};

#endif /* SignRecognition_hpp */
