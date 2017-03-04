//
//  ImageLoader.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ImageLoader_hpp
#define ImageLoader_hpp

#include <stdio.h>
#include <iostream>
//#include <fstream>

#endif /* ImageLoader_hpp */

#define NUMBER_OF_CLASSES 1
#define NUMBER_OF_IMAGES 5000
#define IMAGE_WIDTH 52
#define IMAGE_HEIGHT 52
#define NUMBER_OF_COLORS 3

#define FOLDER_PATH "/Users/pilinszki-nagycsongor/Developer/train-52x52/"
#define HEADER_LENGTH 138

using std::cout;
using std::endl;
using std::string;
using std::to_string;

class ImageLoader {
private:
    unsigned char***** images;
    unsigned char imagePixels[IMAGE_WIDTH * IMAGE_HEIGHT * NUMBER_OF_COLORS];
public:
    ImageLoader();
    ~ImageLoader();
    void loadImages();
    void loadImage(unsigned int, unsigned int);
    string getImagePath(unsigned int, unsigned int);
    unsigned int numberOfDigits(unsigned int);
    unsigned char*** getImageArray(unsigned int, unsigned int);
};
