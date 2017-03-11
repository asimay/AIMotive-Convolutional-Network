//
//  Layer.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 07..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "Layer.hpp"


unsigned int Layer::flatten2DCoordinates(unsigned int x, unsigned int y, unsigned int size) {
    if (x >= size || y >= size) throw "Bad coordinates";
    return x * size + y;
}

unsigned int Layer::flatten3DCoordinates(unsigned int x, unsigned int y, unsigned int z, unsigned int xySize, unsigned int zSize) {
    return flatten2DCoordinates(x, y, xySize) * zSize + z;
}
