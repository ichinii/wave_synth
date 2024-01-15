#pragma once

#include "misc.h"

// *************** misc ***************

__device__
glm::ivec2 index_to_coord(int index) {
    return glm::ivec2(index % W, index / W);
}

// *************** simulate waves ***************

__global__
void step(float* dst, float* src, float* prev_src) {
    const float speed = 0.5f;
    const float dt = 1.0f;
    const float dx = 1.0f;
    static_assert(speed * dt / dx <= 0.5f);

    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto index = [gid] (int offset) { return (gid + offset + N) % N; };
    // auto index = [gid] (int offset) { return gid + offset; };

    float self = src[index(0)];
    float prev_self = prev_src[index(0)];
    float left = src[index(-1)];
    float right = src[index(1)];
    float bottom = src[index(-W)];
    float top = src[index(W)];

    float value = right + left + top + bottom - 4.0f*self;
    value *= speed * dt / dx;
    value += 2.0f*self - prev_self;

    // const float damping = 1.0f;
    const float damping = 0.99f;
    dst[gid] = value * damping;
}

// *************** sources ***************

__global__
void point_source(float* dst, glm::ivec2 p, float t, float amp) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    if (length(glm::vec2(p - coord)) < 10)
        dst[gid] = amp * sin(t * pi2);
}

__global__
void impulse(float* dst, glm::ivec2 p, float amp = 1.0f) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    if (p == coord)
        dst[gid] = amp;
}

// *************** walls ***************

__device__
bool sd_segment(glm::vec2 p, glm::vec2 a, glm::vec2 b, int thickness) {
    glm::vec2 pa = p-a;
    glm::vec2 ba = b-a;
    float h = glm::clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    glm::vec2 v = pa - ba*h;
    return v.x*v.x + v.y*v.y < thickness;
}

__global__
void wall_line(bool* walls, glm::ivec2 from, glm::ivec2 to, int thickness = 2) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    bool inside = sd_segment(coord, from, to, thickness);
    if (inside)
        walls[gid] = true;
}

__global__
void insert_walls(float* dst, bool* walls) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (walls[gid])
        dst[gid] = 0.0f;
}

// *************** draw ***************

__global__
void draw(glm::vec4* output, float* data, bool* walls, Events::Wall wall_event) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    float value = data[gid];
    float high = max(0.0f, value);
    float low = max(0.0f, -value);
    float overflow = max(0.0f, abs(value) - 1.0f);
    glm::vec3 c = glm::vec3(high, low, overflow);

    if (walls[gid])
        c = glm::vec3(1);

    if (wall_event.drawing) {
        auto coord = index_to_coord(gid);
        bool inside = sd_segment(coord, wall_event.to, wall_event.hover, 7);
        if (inside)
            c += 0.01f;
    }

    c = glm::pow(glm::vec3(c), glm::vec3(1.0f/2.2f));

    output[gid] = glm::vec4(c, 1.0f);
}
