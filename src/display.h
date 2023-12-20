#pragma once

#include <functional>

#include <glm/glm.hpp>

extern void display(int width, int height, std::function<glm::vec4*()> update);
