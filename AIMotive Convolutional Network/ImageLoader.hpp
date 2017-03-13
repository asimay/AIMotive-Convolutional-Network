//
//  ImageLoader.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ImageLoader_hpp
#define ImageLoader_hpp

#include <iostream>
#include <Eigen>
#include "Layer3D.hpp"

//A class that loads and stores all the pictures for the convolutional network.
class ImageLoader : public Layer3D {
    
private:
    Layer3D* nextLayer;
    
    int numberOfClasses;
    int numberOfImages;
    int imageSize;
    int numberOfColors;
    
    std::string folderPath;
    
    std::vector<std::vector<Eigen::MatrixXf>> imageMatrices;
    Eigen::MatrixXf outputImage;
    
    //Normalizes the pixel values
    void normalizePixel(unsigned char pixelValue);
    
public:
    ImageLoader(std::string folderPath, int numberOfClasses, int numberOfImages, int imageSize, int numberOfColors);
    ~ImageLoader();
    
    Eigen::MatrixXf* getOutput() { return &outputImage; }
    int getSize() { return imageSize; }
    int getDepth() { return numberOfColors; }
    
    void setPreviousLayer(Layer3D* previousLayer) {}
    void setNextLayer(Layer3D* nextLayer) { this->nextLayer = nextLayer; }
    
    void forwardPropagation() { nextLayer->forwardPropagation(); }
    void backwardPropagation() {}
    
    void loadImages();
    void loadImage(std::string, int, int);
    std::string getImagePath(std::string, int, int);
    int numberOfDigits(int);
    float normalize(float);
};

#endif /* ImageLoader_hpp */
