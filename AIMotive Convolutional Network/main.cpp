//
//  main.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 14..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include <iostream>

#include "SignRecognition.hpp"
#include "ImageLoader.hpp"
#include "ConvolutionLayer.hpp"

int main(int argc, const char * argv[]) {
    
    ImageLoader imageLoader = ImageLoader();
    imageLoader.loadImages();
    
    ConvolutionLayer convLayer(52, 3, 7, 10, 1);
    convLayer.loadImageArray(imageLoader.getImageArray(1, 0));
    convLayer.normalizeInputArray();
    convLayer.flattenInputArray();
    
    
    return 0;
}



