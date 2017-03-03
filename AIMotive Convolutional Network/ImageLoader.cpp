//
//  ImageLoader.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ImageLoader.hpp"

#define NUMBER_OF_CLASSES 12
#define NUMBER_OF_IMAGES 5000
#define IMAGE_WIDTH 52
#define IMAGE_HEIGHT 52
#define NUMBER_OF_COLORS 3

ImageLoader::ImageLoader() {
    images = new unsigned int****[NUMBER_OF_CLASSES];
    for(unsigned int classNumber = 0; classNumber < NUMBER_OF_CLASSES; classNumber++) {         // 1-12
        images[classNumber] = new unsigned int***[NUMBER_OF_IMAGES];
        for (unsigned int imageNumber = 0; imageNumber < NUMBER_OF_IMAGES; imageNumber++) {     // 0-4999
            images[classNumber][imageNumber] = new unsigned int**[IMAGE_WIDTH];
            for (unsigned int x = 0; x < IMAGE_WIDTH; x++) {                                    // 0-51
                images[classNumber][imageNumber][x] = new unsigned int*[IMAGE_HEIGHT];
                for (unsigned int y = 0; y < IMAGE_HEIGHT; y++) {                               // 0-51
                    images[classNumber][imageNumber][x][y] = new unsigned int[NUMBER_OF_COLORS];
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
