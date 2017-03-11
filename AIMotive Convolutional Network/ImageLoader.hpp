//
//  ImageLoader.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ImageLoader_hpp
#define ImageLoader_hpp

#include "Layer.hpp"

#include <iostream>

//A class that loads and stores all the pictures for the convolutional network.
class ImageLoader : public Layer {
    
private:
    Layer* nextLayer;
    
    unsigned int numberOfClasses;
    unsigned int numberOfImages;
    unsigned int imageSize;
    unsigned int numberOfColors;
    
    std::string folderPath;
    
    std::vector<std::vector<Eigen::MatrixXf>> imageMatrices;
    
    //Normalizes the pixel values
    void normalizePixel(unsigned char pixelValue);
    
public:
    ImageLoader(std::string folderPath, unsigned int numberOfClasses, unsigned int numberOfImages, unsigned int imageSize, unsigned int numberOfColors);
    ~ImageLoader();
    
    Eigen::MatrixXf* getOutput();
    unsigned int getOutputSize();
    unsigned int getOutputDepth();
    void setPreviousLayer(Layer* previousLayer);
    void setNextLayer(Layer* nextLayer);
    
    void loadImages();
    void loadImage(std::string, unsigned int, unsigned int);
    std::string getImagePath(std::string, unsigned int, unsigned int);
    unsigned int numberOfDigits(unsigned int);
    float normalize(float);
};

#endif /* ImageLoader_hpp */
