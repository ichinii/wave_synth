#include <iostream>
#include <vector>
#include <chrono>
#include <glm/glm.hpp>
#include "display.h"
#include "misc.h"

constexpr int W = 1024;
constexpr int H = 512;
constexpr int N = W*H;

constexpr int B = 128;
constexpr int G = N/B;

#define pi 3.14159265358979323844f
#define pi2 (pi*2.0f)

__device__
glm::ivec2 index_to_coord(int index) {
    return glm::ivec2(index % W, index / W);
}

__global__
void init(float* src) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    src[gid] = 0.0f;
}

__global__
void point_source(float* dst, glm::ivec2 p, float t, float amp) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    if (length(glm::vec2(p - coord)) < 10)
        dst[gid] = amp * sin(t * pi);
}

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
void impulse(float* dst, glm::ivec2 p, float amp = 1.0f) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    if (p == coord)
        dst[gid] = amp;
}

__global__
void insert_walls(float* dst, bool* walls) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (walls[gid])
        dst[gid] = 0.0f;
}

__global__
void step(float* dst, float* src, float* prev_src) {
    const float dt = 60.0f / 48000.0f;
    const float dx = 1.0f;
    // const float speed = 0.05f;
    const float speed = 1.0f;
    static_assert(speed * dt / dx <= 1.0f);

    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto index = [gid] (int offset) { return (gid + offset + N) % N; };
    // auto index = [gid] (int offset) { return gid + offset; };

    float self = src[index(0)];
    float prev_self = prev_src[index(0)];
    float left = src[index(-1)];
    float right = src[index(1)];
    float bottom = src[index(-W)];
    float top = src[index(W)];

    float next = 2.0f*self - prev_self + (speed * dt / dx) * (right + left - 2.0f*self + top + bottom - 2.0f*self);
    dst[gid] = next;
}

__global__
void draw(glm::vec4* output, float* data, bool* walls) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    float value = data[gid];
    float high = max(0.0f, value);
    float low = max(0.0f, -value);
    float overflow = max(0.0f, abs(value) - 1.0f);
    glm::vec4 c = glm::vec4(high, low, overflow, 1);

    if (walls[gid])
        c = glm::vec4(1);

    output[gid] = c;
}

float probe(float* result, glm::ivec2 coord) {
    float value;
    result += coord.x + coord.y * W;
    cudaMemcpy(&value, result, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

int main() {
    float* dst;
    float* src;
    float* prev_src;
    bool* walls;

    check << cudaMalloc(&dst, N * sizeof(float));
    check << cudaMalloc(&src, N * sizeof(float));
    check << cudaMalloc(&prev_src, N * sizeof(float));
    check << cudaMalloc(&walls, N * sizeof(bool));

    init<<<G, B>>>(src);
    check_kernel();
    check << cudaMemcpy(prev_src, src, N * sizeof(float), cudaMemcpyDeviceToDevice);

    wall_line<<<G, B>>>(walls, glm::ivec2(200, 200), glm::ivec2(400, 200), 10);
    wall_line<<<G, B>>>(walls, glm::ivec2(400, 200), glm::ivec2(400, 400), 10);

    wall_line<<<G, B>>>(walls, glm::ivec2(600, 200), glm::ivec2(800, 200), 1);
    wall_line<<<G, B>>>(walls, glm::ivec2(600, 200), glm::ivec2(600, 300), 1);
    wall_line<<<G, B>>>(walls, glm::ivec2(600, 300), glm::ivec2(800, 300), 1);
    wall_line<<<G, B>>>(walls, glm::ivec2(800, 300), glm::ivec2(800, 210), 1);

    wall_line<<<G, B>>>(walls, glm::ivec2(500, 50), glm::ivec2(900, 90), 1);

    {
        int w = W-1;
        int h = H-1;
        wall_line<<<G, B>>>(walls, glm::ivec2(0, 0), glm::ivec2(w, 0), 1); check_kernel();
        wall_line<<<G, B>>>(walls, glm::ivec2(w, 0), glm::ivec2(w, h), 1); check_kernel();
        wall_line<<<G, B>>>(walls, glm::ivec2(w, h), glm::ivec2(0, h), 1); check_kernel();
        wall_line<<<G, B>>>(walls, glm::ivec2(0, h), glm::ivec2(0, 0), 1); check_kernel();
    }

    {
        glm::vec4* d_output;
        check << cudaMalloc(&d_output, N * sizeof(glm::vec4));

        auto h_output = std::vector<glm::vec4>(N);

        using namespace std::chrono;

        display(W, H, [&] (ClickEvent ev) {
            static auto start_time = steady_clock::now();
            static auto frame = 0ul;

            for (int i = 0; i < 48000 / 60; ++i) {
                auto t = frame / 48000.0f;

                // auto time = steady_clock::now();
                // auto t = duration_cast<milliseconds>(time - start_time).count() / 1000.0f;

                if (false && frame < 48000) {
                    point_source<<<G, B>>>(
                        src,
                        glm::ivec2(100, 100),
                        t * 55.0f,
                        pow(1.0f - min(1.0f, frame / 48000.0f), 2.0f)
                    );
                    check_kernel();
                }

                // point_source<<<G, B>>>(src, glm::ivec2(100, 100), t*440.0f, 0.5f);
                // check_kernel();

                insert_walls<<<G, B>>>(src, walls);
                check_kernel();

                if (ev.clicked) {
                    // impulse<<<G, B>>>(src, glm::ivec2(ev.x, H - ev.y), 0.5f);
                    point_source<<<G, B>>>(src, glm::ivec2(ev.x, H-ev.y), t*440.0f, 0.5f);
                    check_kernel();
                }

                static glm::ivec2 from = glm::ivec2(0, 0);
                if (ev.clocked) {
                    auto to = glm::ivec2(ev.x, H-ev.y);
                    if (from != glm::ivec2{0, 0}) {
                        wall_line<<<G, B>>>(walls, from, to, 1); check_kernel();
                        check_kernel();
                    }
                    from = to;
                }

                step<<<G, B>>>(dst, src, prev_src);
                check_kernel();

                auto tmp = prev_src;
                prev_src = src;
                src = dst;
                dst = tmp;

                ++frame;
            }

            auto result = src;

            draw<<<G, B>>>(d_output, result, walls);
            check_kernel();

            check << cudaMemcpy(h_output.data(), d_output, N * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

            // float amp = probe(result, glm::ivec2(50, 50));
            // std::cout << amp << std::endl;

            return h_output.data();
        });

        check << cudaFree(d_output);
    }

    check << cudaFree(src);
    check << cudaFree(dst);

    return 0;
}
