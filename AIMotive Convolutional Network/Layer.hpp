//
//  Layer.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 07..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <Eigen>

class Layer {
public:
    Layer() {}
    virtual ~Layer() {}
    virtual Eigen::MatrixXf* getOutput() = 0;
    virtual unsigned int getOutputSize() = 0;
    virtual unsigned int getOutputDepth() = 0;
    virtual void setPreviousLayer(Layer* previousLayer) = 0;
    virtual void setNextLayer(Layer* nextLayer) = 0;
    static unsigned int flatten2DCoordinates(unsigned int x, unsigned int y, unsigned int size);
    static unsigned int flatten3DCoordinates(unsigned int x, unsigned int y, unsigned int z, unsigned int xySize, unsigned int zSize);
};

#endif /* Layer_hpp */
