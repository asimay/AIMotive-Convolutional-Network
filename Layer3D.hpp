//
//  Layer3D.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 11..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef Layer3D_hpp
#define Layer3D_hpp

#include <stdio.h>
#include <Eigen>

class Layer3D {
    
public:
    Layer3D() {}
    virtual ~Layer3D() {}
    virtual Eigen::MatrixXf* getOutput() = 0;
    virtual unsigned int getSize() = 0;
    virtual unsigned int getDepth() = 0;
    
    virtual void setPreviousLayer(Layer3D* previousLayer) = 0;
    virtual void setNextLayer(Layer3D* nextLayer) = 0;
    
    virtual void forwardPropagation() = 0;
    virtual void backwardPropagation() = 0;
    
    static unsigned int flatten2DCoordinates(unsigned int x, unsigned int y, unsigned int size);
    static unsigned int flatten3DCoordinates(unsigned int x, unsigned int y, unsigned int z, unsigned int xySize, unsigned int zSize);
    
};

#endif /* Layer3D_hpp */
