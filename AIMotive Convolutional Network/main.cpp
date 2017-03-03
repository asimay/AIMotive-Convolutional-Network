//
//  main.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 14..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include <iostream>

#include "SignRecognition.hpp"

int main(int argc, const char * argv[]) {
    
    SignRecognition signRecognition;
    signRecognition.loadImage(1, 0);
    
    return 0;
}



