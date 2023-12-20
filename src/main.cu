#include <iostream>
#include <vector>
#include <chrono>
#include <glm/glm.hpp>
#include "display.h"
#include "misc.h"

constexpr int N = 1024;
constexpr int B = 128;
constexpr int G = N/B;

__global__
void init(float* src) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    // src[gid] = (gid+N/8 < N/4) ? 0.4f * sin(gid / float(N/8) * 3.141f * 2.0f) : 0.0f;
    // src[gid] = gid == N/2 ? 1.0f : 0.0f;
    src[gid] = 0.0f;
}

__global__
void point_source(float* dst, float x, float t, float amp) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    if (gid == x)
        dst[gid] = amp * sin(t);
}

__global__
void step(float* dst, float* src, float* prev_src) {
    // const float dt = 0.95f;
    // const float dx = 1.0f;
    // const float speed = 1.0f;

    int width = blockDim.x*gridDim.x;
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto index = [gid, width] (int offset) { return (gid + offset + width) % width; };

    float self = src[index(0)];
    float prev_self = prev_src[index(0)];

    float left = src[index(-1)];
    float right = src[index(1)];

    float next = 2.0f*self - prev_self + (right + left - 2.0f*self);

    dst[gid] = next;
}

__global__
void draw(glm::vec4* output, float* data) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;

    int width = N;
    int height = blockDim.x*gridDim.x / width;
    float gidy = static_cast<float>(gid / width);
    float y = gidy / height * 2.0f - 1.0f;

    int index = gid % N;
    float value = data[index];

    // bool hit = abs(y - value) < 0.01f;
    bool hit = 0.0f <= y && y <= value
        || y < 0.0f && value <= y;
    output[gid] = glm::vec4(hit, gidy == height/2, 0, 1);
}

int main() {
    float* dst;
    float* src;
    float* prev_src;

    check << cudaMalloc(&dst, N * sizeof(float));
    check << cudaMalloc(&src, N * sizeof(float));
    check << cudaMalloc(&prev_src, N * sizeof(float));

    init<<<G, B>>>(src);
    check_kernel();
    check << cudaMemcpy(prev_src, src, N * sizeof(float), cudaMemcpyDeviceToDevice);

    {
        int width = N;
        int height = 256;
        int S = width*height;

        glm::vec4* d_output;
        check << cudaMalloc(&d_output, S * sizeof(glm::vec4));

        auto h_output = std::vector<glm::vec4>(S);

        using namespace std::chrono;

        display(width, height, [&] {
            static auto start_time = steady_clock::now();
            auto time = steady_clock::now();
            auto t = duration_cast<milliseconds>(time - start_time).count() / 1000.0f;
            point_source<<<G, B>>>(src, N/4, t*2.0f + sin(t), 0.2f);

            step<<<G, B>>>(dst, src, prev_src);
            check_kernel();

            draw<<<G * height, B>>>(d_output, dst);
            check_kernel();

            check << cudaMemcpy(h_output.data(), d_output, S * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

            auto tmp = prev_src;
            prev_src = src;
            src = dst;
            dst = tmp;

            return h_output.data();
        });

        check << cudaFree(d_output);
    }

    check << cudaFree(src);
    check << cudaFree(dst);

    return 0;
}
