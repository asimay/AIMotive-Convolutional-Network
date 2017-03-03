//
//  ImageLoader.hpp
//  AIMotive Convolutional Network
//
//  Created by Pilinszki-Nagy Csongor on 2017. 03. 03..
//  Copyright © 2017. Csongor Pilinszki-Nagy. All rights reserved.
//

#ifndef ImageLoader_hpp
#define ImageLoader_hpp

#include <stdio.h>
#include <iostream>

#endif /* ImageLoader_hpp */

using std::cout;
using std::endl;

class ImageLoader {
private:
    unsigned char***** images;
    unsigned char info[138];
public:
    ImageLoader();
    ~ImageLoader();
    void loadImages();
};
