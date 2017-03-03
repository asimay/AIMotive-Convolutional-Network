//
//  SignRecognition.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 28..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef SignRecognition_hpp
#define SignRecognition_hpp

#include <stdio.h>
#include <iostream>
#include <Eigen>

#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"
#define HEADER_LENGTH 138

#define IMAGE_HEIGHT 52
#define IMAGE_WIDTH 52
#define IMAGE_DEPTH 3

#define FILTER_1_HEIGHT 3
#define FILTER_1_WIDTH 3
#define FILTER_1_DEPTH 3
#define FILTER_1_FLATTEN FILTER_1_HEIGHT * FILTER_1_WIDTH * FILTER_1_DEPTH
#define FILTER_1_NUMBER 10

#endif /* SignRecognition_hpp */

class SignRecognition {
private:
    
public:
    SignRecognition();
    ~SignRecognition();
};
