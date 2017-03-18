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
    
    static int flatten2DCoordinates(int x, int y, int size);
    static int flatten3DCoordinates(int x, int y, int z, int xySize, int zSize);
    
};

#endif /* Layer3D_hpp */
