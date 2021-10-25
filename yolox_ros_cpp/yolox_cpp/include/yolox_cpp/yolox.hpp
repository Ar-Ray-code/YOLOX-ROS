#ifndef _YOLOX_CPP_YOLOX_HPP
#define _YOLOX_CPP_YOLOX_HPP

#include "config.h"

#ifdef YOLOX_USE_OPENVINO
    #include "yolox_openvino.hpp"
#endif

#ifdef YOLOX_USE_TENSORRT
    #include "yolox_tensorrt.hpp"
#endif


#endif