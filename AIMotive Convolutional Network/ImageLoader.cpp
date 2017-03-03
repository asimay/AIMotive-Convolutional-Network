//
//  ImageLoader.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ImageLoader.hpp"


ImageLoader::ImageLoader() {
    images = new unsigned char****[NUMBER_OF_CLASSES];
    for(unsigned int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {         // 1-12
        images[classNumber] = new unsigned char***[NUMBER_OF_IMAGES];
        for (unsigned int imageNumber = 0; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {     // 0-4999
            images[classNumber][imageNumber] = new unsigned char**[IMAGE_WIDTH];
            for (unsigned int x = 0; x < IMAGE_WIDTH; x++) {                                    // 0-51
                images[classNumber][imageNumber][x] = new unsigned char*[IMAGE_HEIGHT];
                for (unsigned int y = 0; y < IMAGE_HEIGHT; y++) {                               // 0-51
                    images[classNumber][imageNumber][x][y] = new unsigned char[NUMBER_OF_COLORS];
                }
            }
        }
    }
    cout << "ImageLoader created" << endl;
}

ImageLoader::~ImageLoader() {
    for (unsigned int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {
        for (unsigned int imageNumber = 0; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {
            for (unsigned int x = 0; x < IMAGE_WIDTH; x++) {
                for (unsigned int y = 0; y < IMAGE_HEIGHT; y++) {
                    delete [] images[classNumber][imageNumber][x][y];
                }
                delete [] images[classNumber][imageNumber][x];
            }
            delete [] images[classNumber][imageNumber];
        }
        delete [] images[classNumber];
    }
    delete [] images;
    cout << "ImageLoader deleted" << endl;
}

void ImageLoader::loadImages() {
    for (unsigned int classNumber = 1; classNumber <= NUMBER_OF_CLASSES; classNumber++) {
        for (unsigned int imageNumber = 0; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {
            loadImage(classNumber, imageNumber);
            cout << "Class: " << classNumber << " Image: " << imageNumber << " LOADED" << endl;
        }
    }
}

void ImageLoader::loadImage(unsigned int classNumber, unsigned int imageNumber) {
    string imagePath = getImagePath(classNumber, imageNumber);
    
    cout << imagePath << endl;
    
    FILE* inputImage = fopen(imagePath.c_str(), "rb");
    
    if(inputImage == NULL)
        throw "File not found.";
    
    unsigned char fileHeader[HEADER_LENGTH];
    fread(fileHeader, sizeof(unsigned char), HEADER_LENGTH, inputImage);
    
    fread(imagePixels, sizeof(unsigned char), IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_COLORS, inputImage);
    
    for (unsigned int x = 0; x < IMAGE_WIDTH; x++) {
        for (unsigned int y = 0; y < IMAGE_HEIGHT; y++) {
            for (unsigned int color = 0; color < NUMBER_OF_COLORS; color++) {
                images[classNumber-1][imageNumber][x][y][color] = imagePixels[x * NUMBER_OF_COLORS + y * IMAGE_WIDTH * NUMBER_OF_COLORS + color];
            }
        }
    }
    
    fclose(inputImage);
}

string ImageLoader::getImagePath(unsigned int classNumber, unsigned int imageNumber) {
    string imagePath = FOLDER_PATH;
    imagePath += to_string(classNumber);
    imagePath += "/";
    imagePath += to_string(classNumber);
    imagePath += "_";
    
    unsigned int paddingZeros = 4 - numberOfDigits(imageNumber);
    for (unsigned int i = 0; i < paddingZeros; i++) {
        imagePath += "0";
    }
    
    imagePath += to_string(imageNumber);
    imagePath += ".bmp";
    
    return imagePath;
}

unsigned int ImageLoader::numberOfDigits(unsigned int number) {
    if (number == 0) return 1;
    unsigned int numberOfDigits = 0;
    while (number) {
        numberOfDigits++;
        number /= 10;
    }
    return numberOfDigits;
}
