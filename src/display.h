#pragma once

#include <functional>
#include <map>

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

using KeyState = std::map<int, int>;
using MouseState = std::map<int, int>;
using MousePos = glm::ivec2;
using MouseScroll = glm::vec2;

struct InputState {
    KeyState keys;
    MouseState mouse;
    MousePos mouse_pos;
    MouseScroll mouse_scroll;
};

inline bool key_pressed(const InputState& input, int key) {
    auto it = input.keys.find(key);
    if (it != input.keys.end())
        return it->second > 0;
    return false;
}

inline bool key_just_pressed(const InputState& input, int key) {
    auto it = input.keys.find(key);
    if (it != input.keys.end())
        return it->second == 1;
    return false;
}

inline bool button_pressed(const InputState& input, int button) {
    auto it = input.mouse.find(button);
    if (it != input.mouse.end())
        return it->second > 0;
    return false;
}

using UpdateFn = std::function<glm::vec4*(const InputState&)>;
extern void display(int width, int height, UpdateFn update);
