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
#include "Layer3D.hpp"
#include <iostream>

class ConvolutionLayer : public Layer3D {
    
private:
    
    Layer3D* previousLayer;
    Layer3D* nextLayer;
    
    int imageSize;
    int filterSize;
    int filterNumber;
    int stride;
    
    Eigen::MatrixXf filterMatrix;
    Eigen::MatrixXf inputMatrix;
    Eigen::MatrixXf outputMatrix;
    Eigen::MatrixXf deltaInputMatrix;
    Eigen::MatrixXf deltaOutputMatrix;
    
public:
    
    ConvolutionLayer(int filterSize, int filterNumber, int stride);
    ~ConvolutionLayer();
    
    static Eigen::MatrixXf flattenMatrix(Eigen::MatrixXf* inputMatrix, int inputSize, int inputDepth, int filterSize, int stride);
    static Eigen::VectorXf flattenReceptiveField(Eigen::MatrixXf* inputMatrix, int inputSize, int inputDepth, int inputX, int inputY, int filterSize);
    static void addBiasColumn(Eigen::MatrixXf* inputMatrix);
    static Eigen::MatrixXf reorderMatrix(Eigen::MatrixXf* inputMatrix, int inputSize, int inputDepth, int filterSize, int stride);
    static void reorderReceptiveField(Eigen::MatrixXf* inputMatrix, Eigen::MatrixXf* outputMatrix, int inputSize, int inputDepth, int inputX, int inputY, int filterSize, int stride);
    
    int getSize() { return imageSize; }
    int getDepth() { return filterNumber; }
    int getFilterSize() { return filterSize; }
    int getFilterNumber() { return filterNumber; }
    int getStride() { return stride; }
    
    
    Eigen::MatrixXf* getFilter() { return &filterMatrix; }
    Eigen::MatrixXf* getInput() { return &inputMatrix; }
    Eigen::MatrixXf* getOutput() { return &outputMatrix; }
    Eigen::MatrixXf* getDeltaInput() { return &deltaInputMatrix; }
    Eigen::MatrixXf* getDeltaOutput() { return &deltaOutputMatrix; }
    
    void setPreviousLayer(Layer3D* previousLayer) {
        this->previousLayer = previousLayer;
        //previousLayer->setNextLayer(this);
        //imageSize = (previousLayer->getSize() - 1) / stride + 1;
        //filterMatrix = Eigen::MatrixXf::Random(filterSize * filterSize * previousLayer->getDepth() + 1, filterNumber);
    }
    
    void setNextLayer(Layer3D* nextLayer) {
        this->nextLayer = nextLayer;
    }
    
    void forwardPropagation();
    void backwardPropagation();
    
};



#endif /* ConvolutionLayer_hpp */
