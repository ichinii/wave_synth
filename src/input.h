#pragma once

#include <map>

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "common.h"

// *************** input ***************

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

// *************** translate input to events ***************

struct Events {
    bool clear_waves = false;
    bool clear_walls = false;

    struct PointSource {
        float active = false;
        float freq;
        glm::vec2 pos;
    } point_source;

    struct Wall {
        bool place = false;
        bool drawing = false;
        glm::vec2 from;
        glm::vec2 to;
        glm::vec2 hover;
    } wall;
};

inline Events input_state_to_events(const InputState& input, Events events) {
    events.clear_waves = key_just_pressed(input, GLFW_KEY_C);
    events.clear_walls = key_just_pressed(input, GLFW_KEY_V);

    {
        float freq = 0.0f;
        freq = key_pressed(input, GLFW_KEY_1) ? 1.0f : freq;
        freq = key_pressed(input, GLFW_KEY_2) ? 2.0f : freq;
        freq = key_pressed(input, GLFW_KEY_3) ? 3.0f : freq;
        freq = key_pressed(input, GLFW_KEY_4) ? 4.0f : freq;
        freq = key_pressed(input, GLFW_KEY_5) ? 5.0f : freq;
        freq = key_pressed(input, GLFW_KEY_6) ? 6.0f : freq;
        freq = key_pressed(input, GLFW_KEY_7) ? 7.0f : freq;
        freq = key_pressed(input, GLFW_KEY_8) ? 8.0f : freq;
        freq = key_pressed(input, GLFW_KEY_9) ? 9.0f : freq;

        events.point_source.active = freq > 0.0f;
        events.point_source.freq = freq * ProcessedSamplesPerAudioBuffer;
        events.point_source.pos = input.mouse_pos;
    }

    {
        bool pressed = button_pressed(input, GLFW_MOUSE_BUTTON_LEFT);
        bool drawing = events.wall.drawing || pressed;
        bool place = events.wall.drawing && pressed;

        bool reset = button_pressed(input, GLFW_MOUSE_BUTTON_RIGHT)
            || events.point_source.active;
        place = !reset && place;
        drawing = !reset && drawing;

        if (pressed) {
            events.wall.from = events.wall.to;
            events.wall.to = glm::vec2(input.mouse_pos);
        }

        events.wall.hover = glm::vec2(input.mouse_pos);
        events.wall.place = place;
        events.wall.drawing = drawing;
    }

    return events;
}
