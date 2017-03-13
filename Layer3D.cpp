//
//  Layer3D.cpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 11..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#include "Layer3D.hpp"

int Layer3D::flatten2DCoordinates(int x, int y, int size) {
    if (x >= size || y >= size) throw "Bad coordinates";
    return x * size + y;
}

int Layer3D::flatten3DCoordinates(int x, int y, int z, int xySize, int zSize) {
    return flatten2DCoordinates(x, y, xySize) * zSize + z;
}
