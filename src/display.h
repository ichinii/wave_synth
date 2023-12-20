#pragma once

#include <functional>

#include <glm/glm.hpp>

struct ClickEvent {
    bool clicked;
    bool clocked;
    int x;
    int y;
};

using UpdateFn = std::function<glm::vec4*(ClickEvent)>;
extern void display(int width, int height, UpdateFn update);
