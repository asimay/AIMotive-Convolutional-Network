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
    
    const int numberOfClasses;
    const int numberOfImages;
    const int imageSize;
    const int numberOfColors;
    
    std::vector<std::vector<Eigen::MatrixXf>> imageMatrices;
    
public:
    ImageLoader() : numberOfClasses(0), numberOfImages(0), imageSize(0), numberOfColors(0), imageMatrices(std::vector<std::vector<Eigen::MatrixXf>>()) {}
    ImageLoader(int numberOfClasses, int numberOfImages, int imageSize, int numberOfColors) : numberOfClasses(numberOfClasses), numberOfImages(numberOfImages), imageSize(imageSize), numberOfColors(numberOfColors) {
        imageMatrices = std::vector<std::vector<Eigen::MatrixXf>>(numberOfClasses, std::vector<Eigen::MatrixXf>(numberOfImages, Eigen::MatrixXf()));
    }
    ~ImageLoader() {}
    
    void loadImages(std::string floderPath);
    Eigen::MatrixXf loadImage(std::string folderPath, int classNumber, int imageNumber);
    static std::string getImagePath(std::string folderPath, int classNumber, int imageNumber);
    static float normalize(float number);
    static int numberOfDigits(int number);
    Eigen::MatrixXf getImageMatrix(int imageClass, int imageNumber) { return imageMatrices[imageClass][imageNumber]; }

};

#endif /* ImageLoader_hpp */
