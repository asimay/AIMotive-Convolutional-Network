//
//  SignRecognition.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 02. 28..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "SignRecognition.hpp"

SignRecognition::SignRecognition() {
}

SignRecognition::~SignRecognition() {
}

/*void SignRecognition::loadImage(unsigned int signClass, unsigned int imageNumber) {
    
    string filePath = FOLDER_PATH;
    filePath += to_string(signClass);
    filePath += "/";
    filePath += to_string(signClass);
    filePath += "_";
    
    unsigned int paddingZeros = 4 - numberOfDigits(imageNumber);
    for (unsigned int i = 0; i < paddingZeros; i++) {
        filePath += "0";
    }
    
    filePath += to_string(imageNumber);
    filePath += ".bmp";
    
    cout << filePath << endl;
    
    FILE* inputFile = fopen(filePath.c_str(), "rb");
    
    if(inputFile == NULL)
        throw "File not found.";
    
    unsigned char fileHeader[HEADER_LENGTH];
    fread(fileHeader, sizeof(unsigned char), HEADER_LENGTH, inputFile);
    
    unsigned char* pixelData = new unsigned char[IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH];
    fread(pixelData, sizeof(unsigned char), IMAGE_WIDTH * IMAGE_WIDTH * IMAGE_DEPTH, inputFile);
    
    for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH; i++) {
        unsigned int row = i / (IMAGE_WIDTH * IMAGE_DEPTH);
        unsigned int column = i % (IMAGE_WIDTH * IMAGE_DEPTH) / IMAGE_DEPTH;
        unsigned int color = i % IMAGE_DEPTH; //0 BLUE, 1 GREEN, 2 RED
        
        inputImage[row][column][color] = (unsigned int)pixelData[i];
    }
    cout << inputImage[0][0][2] << " " << inputImage[0][0][1] << " " << inputImage[0][0][0] << " " << endl;
    cout << inputImage[0][51][2] << " " << inputImage[0][51][1] << " " << inputImage[0][51][0] << " " << endl;
    cout << inputImage[51][0][2] << " " << inputImage[51][0][1] << " " << inputImage[51][0][0] << " " << endl;
    cout << inputImage[51][51][2] << " " << inputImage[51][51][1] << " " << inputImage[51][51][0] << " " << endl;
}

unsigned int SignRecognition::numberOfDigits(unsigned int number) {
    if (number == 0) return 1;
    unsigned int numberOfDigits = 0;
    while (number) {
        numberOfDigits++;
        number /= 10;
    }
    return numberOfDigits;
}

void SignRecognition::flattenImage() {
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            for (int k = 0; k < FILTER_1_HEIGHT; k++) {
                for (int l = 0; l < FILTER_1_WIDTH; l++) {
                    for (int m = 0; m < FILTER_1_DEPTH; m++) {
                        //unsigned int matrixIndex1 = m * FILTER_1_HEIGHT * FILTER_1_WIDTH + i * FILTER_1_WIDTH + j,
                    }
                }
            }
        }
    }
}
 */

