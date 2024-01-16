#pragma once

#include <functional>

#include <glm/glm.hpp>

#include "input.h"

using UpdateFn = std::function<glm::vec4*(const InputState&)>;
extern void display(int width, int height, UpdateFn update);
