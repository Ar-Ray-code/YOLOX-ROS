#ifndef BBOXES_FROM_YAML_HPP
#define BBOXES_FROM_YAML_HPP

#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

typedef struct
{
    float r;
    float g;
    float b;
} color;

typedef struct
{
    std::string name;
    color rgb;
} coco;

class bboxes_from_yaml
{
public:
    bool load_yaml(std::string filename)
    {
        std::ifstream file(filename);
        coco output;
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        auto node = YAML::Load(file);
        file.close();

        auto class_info = node["class_info"];
        for (auto it = class_info.begin(); it != class_info.end(); it++)
        {
            std::cout << it->first.as<std::string>();
            // get color
            auto color_node = it->second;
            color bbox_color;
            bbox_color = {color_node.as<std::vector<float>>()[0],
                          color_node.as<std::vector<float>>()[1],
                          color_node.as<std::vector<float>>()[2]};

            coco data;
            data.name = it->first.as<std::string>();
            data.rgb.r = bbox_color.r;
            data.rgb.g = bbox_color.g;
            data.rgb.b = bbox_color.b;
            coco_data.push_back(data);
        }
        return true;
    }
    std::vector<coco> coco_data;
};

#endif