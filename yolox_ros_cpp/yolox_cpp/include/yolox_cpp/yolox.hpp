#ifndef _YOLOX_CPP_YOLOX_HPP
#define _YOLOX_CPP_YOLOX_HPP

#include "config.h"

#ifdef ENABLE_OPENVINO
    #include "yolox_openvino.hpp"
#endif

#ifdef ENABLE_TENSORRT
    #include "yolox_tensorrt.hpp"
#endif

#ifdef ENABLE_ONNXRUNTIME
    #include "yolox_onnxruntime.hpp"
#endif

#ifdef ENABLE_TFLITE
    #include "yolox_tflite.hpp"
#endif


#endif
