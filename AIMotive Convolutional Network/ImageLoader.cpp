//
//  ImageLoader.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "ImageLoader.hpp"

#define HEADER_LENGTH 138

ImageLoader::ImageLoader(std::string folderPath, int numberOfClasses, int numberOfImages, int imageSize, int numberOfColors) {
    this->folderPath = folderPath;
    this->numberOfClasses = numberOfClasses;
    this->numberOfImages = numberOfImages;
    this->imageSize = imageSize;
    this->numberOfColors = numberOfColors;
    imageMatrices = std::vector<std::vector<Eigen::MatrixXf>>(numberOfClasses, std::vector<Eigen::MatrixXf>(numberOfImages, Eigen::MatrixXf(imageSize*imageSize, numberOfColors)));
    
    loadImages();
}

ImageLoader::~ImageLoader() {
    

}


void ImageLoader::loadImages() {
    for (int classNumber = 0; classNumber < numberOfClasses; classNumber++) {
        for (int imageNumber = 0; imageNumber < numberOfImages; imageNumber++) {
            loadImage(folderPath, classNumber + 1, imageNumber);
        }
        std::cout << "Class " << classNumber + 1 << " loaded" << std::endl;
    }
}

void ImageLoader::loadImage(std::string folderPath, int classNumber, int imageNumber) {
    std::string imagePath = getImagePath(folderPath, classNumber, imageNumber);
    
    FILE* inputImage = fopen(imagePath.c_str(), "rb");
    
    if(inputImage == NULL)
        throw "File not found.";
    
    unsigned char fileHeader[HEADER_LENGTH];
    fread(fileHeader, sizeof(unsigned char), HEADER_LENGTH, inputImage);
    

    unsigned char imagePixels[imageSize * imageSize * numberOfColors];
    fread(imagePixels, sizeof(unsigned char), imageSize * imageSize * numberOfColors, inputImage);
    
    for (int x = 0; x < imageSize; x++) {
        for (int y = 0; y < imageSize; y++) {
            for (int color = 0; color < numberOfColors; color++) {
                imageMatrices[classNumber - 1][imageNumber](flatten2DCoordinates(x, y, imageSize), color) = normalize ((float) imagePixels[flatten3DCoordinates(x, y, color, imageSize, numberOfColors)]);

            }
        }
    }
    
    fclose(inputImage);
}

std::string ImageLoader::getImagePath(std::string folderPath, int classNumber, int imageNumber) {
    std::string imagePath = folderPath;
    imagePath += std::to_string(classNumber);
    imagePath += "/";
    imagePath += std::to_string(classNumber);
    imagePath += "_";
    
    int paddingZeros = 4 - numberOfDigits(imageNumber);
    for (int i = 0; i < paddingZeros; i++) {
        imagePath += "0";
    }
    
    imagePath += std::to_string(imageNumber);
    imagePath += ".bmp";
    
    return imagePath;
}

int ImageLoader::numberOfDigits(int number) {
    if (number == 0) return 1;
    int numberOfDigits = 0;
    while (number) {
        numberOfDigits++;
        number /= 10;
    }
    return numberOfDigits;
}

float ImageLoader::normalize(float number) {
    return (number - 128.0) / 128.0;
}
