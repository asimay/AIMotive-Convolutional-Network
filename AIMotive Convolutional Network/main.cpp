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


int main(int argc, const char * argv[]) {
    srand((unsigned int)time(NULL));

    SignRecognition::performSimpleRecognition();
    
    return 0;
    
}



