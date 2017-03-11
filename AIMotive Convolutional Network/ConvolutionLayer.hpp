//
//  ConvolutionLayer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ConvolutionLayer_hpp
#define ConvolutionLayer_hpp

#include <Eigen>
#include "Layer.hpp"
#include <iostream>

class ConvolutionLayer : public Layer {
private:
    Layer* previousLayer;
    Layer* nextLayer;
    
    unsigned int imageSize;
    unsigned int filterSize;
    unsigned int filterNumber;
    unsigned int stride;
    
    Eigen::MatrixXf inputMatrix;
    Eigen::MatrixXf filterMatrix;
    Eigen::MatrixXf outputMatrix;
    Eigen::MatrixXf deltaInputMatrix;
    Eigen::MatrixXf deltaOutputMatrix;
    
public:
    ConvolutionLayer(unsigned int filterSize, unsigned int filterNumber, unsigned int stride);
    ~ConvolutionLayer();
    
    Eigen::MatrixXf* getOutput();
    unsigned int getOutputSize();
    unsigned int getOutputDepth();
    void setPreviousLayer(Layer* previousLayer);
    void setNextLayer(Layer* nextLayer);
    
    void forwardConvolution();
    Eigen::MatrixXf getFilterMatrix();
    Eigen::MatrixXf getInputMatrix();
    
    static Eigen::MatrixXf flattenMatrix(Eigen::MatrixXf* inputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int filterSize, unsigned int stride);
    static Eigen::VectorXf flattenReceptiveField(Eigen::MatrixXf* inputMatrix, unsigned int inputSize, unsigned int inputDepth, unsigned int inputX, unsigned int inputY, unsigned int filterSize);
    static void addBiasColumn(Eigen::MatrixXf* inputMatrix);
};



#endif /* ConvolutionLayer_hpp */
