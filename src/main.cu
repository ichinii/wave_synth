#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>
#include <jack/jack.h>

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
void point_source(float* dst, glm::ivec2 p, float t, float amp) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    auto coord = index_to_coord(gid);
    if (length(glm::vec2(p - coord)) < 10)
        dst[gid] = amp * sin(t * pi2);
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

    const float damping = 1.0f;
    // const float damping = 0.999f;
    dst[gid] = value * damping;
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

float probe(float* data, glm::ivec2 coord) {
    float value;
    data += coord.x + coord.y * W;
    cudaMemcpy(&value, data, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void init_walls(bool* d_walls) {
    wall_line<<<G, B>>>(d_walls, glm::ivec2(200, 200), glm::ivec2(400, 200), 10);
    wall_line<<<G, B>>>(d_walls, glm::ivec2(400, 200), glm::ivec2(400, 400), 10);

    wall_line<<<G, B>>>(d_walls, glm::ivec2(600, 200), glm::ivec2(800, 200), 1);
    wall_line<<<G, B>>>(d_walls, glm::ivec2(600, 200), glm::ivec2(600, 300), 1);
    wall_line<<<G, B>>>(d_walls, glm::ivec2(600, 300), glm::ivec2(800, 300), 1);
    wall_line<<<G, B>>>(d_walls, glm::ivec2(800, 300), glm::ivec2(800, 210), 1);

    wall_line<<<G, B>>>(d_walls, glm::ivec2(500, 50), glm::ivec2(900, 90), 1);

    {
        int w = W-1;
        int h = H-1;
        wall_line<<<G, B>>>(d_walls, glm::ivec2(0, 0), glm::ivec2(w, 0), 1); check_kernel();
        wall_line<<<G, B>>>(d_walls, glm::ivec2(w, 0), glm::ivec2(w, h), 1); check_kernel();
        wall_line<<<G, B>>>(d_walls, glm::ivec2(w, h), glm::ivec2(0, h), 1); check_kernel();
        wall_line<<<G, B>>>(d_walls, glm::ivec2(0, h), glm::ivec2(0, 0), 1); check_kernel();
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    std::cout << "hello instrument!" << std::endl;

    float* d_grid;
    float* d_grid_prev;
    float* d_grid_next;
    check << cudaMalloc(&d_grid, N * sizeof(float));
    check << cudaMalloc(&d_grid_prev, N * sizeof(float));
    check << cudaMalloc(&d_grid_next, N * sizeof(float));
    check << cudaMemset(d_grid, 0, N * sizeof(float));
    check << cudaMemset(d_grid_prev, 0, N * sizeof(float));

    bool* d_walls;
    check << cudaMalloc(&d_walls, N * sizeof(bool));
    check << cudaMemset(d_walls, 0, N * sizeof(float));
    init_walls(d_walls);

    auto h_output = std::vector<glm::vec4>(N);
    glm::vec4* d_output;
    check << cudaMalloc(&d_output, N * sizeof(glm::vec4));

    auto swap_buffers = [&] {
        auto tmp = d_grid_prev;
        d_grid_prev = d_grid;
        d_grid = d_grid_next;
        d_grid_next = tmp;
        return d_grid; // return the results buffer
    };

    constexpr int AudioSampleRate = 48000;
    constexpr int AudioBufferSize = 128;
    constexpr int ProbeStepSize = AudioBufferSize;

    // constexpr int AudioBufferPeriods = 2;
    constexpr int AudioBuffersPerSecond = AudioSampleRate / AudioBufferSize;
    static_assert(AudioBuffersPerSecond * AudioBufferSize == AudioSampleRate);

    // the audio buffer size must be a multiple of sample step size
    static_assert(AudioBufferSize / ProbeStepSize > 0);
    static_assert(AudioBufferSize % ProbeStepSize == 0);

    constexpr int StepsPerSecond = (AudioBufferSize / ProbeStepSize) * AudioBuffersPerSecond;
    constexpr float SecondsPerStep = 1.0f / StepsPerSecond;
    // constexpr float PitchCorrectionFactor = AudioSampleRate;

    using namespace std::chrono;
    using namespace std::chrono_literals;
    auto start_time = steady_clock::now();
    auto prev_step_time = 0ms;
    auto steps = 0;

    glm::ivec2 wall_from = glm::ivec2(0, 0);

    float prev_amp = 0.0f;
    float buffer[AudioBufferSize];
    int buffer_index = 0;

    jack_status_t status;
    auto client = jack_client_open("wavy_synth", JackNoStartServer, &status);
    std::cout << "jack status: " << (status ? "error" : "ok") << std::endl;
    assert(!status);
    auto input_port = jack_port_register(
        client,
        "inputxx",
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortIsInput,
        AudioBufferSize
    );
    auto output_port = jack_port_register(
        client,
        "outputxx",
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortIsOutput,
        AudioBufferSize
    );
    assert(input_port);
    assert(output_port);
    struct Ports {
        jack_port_t* input;
        jack_port_t* output;
        float* buffer;
    } ports {
        input_port,
        output_port,
        buffer,
    };
    auto process = [] (jack_nframes_t nframes, void* arg) -> int {
        assert(nframes == AudioBufferSize);
        auto ports = static_cast<Ports*>(arg);
        float* input = static_cast<float*>(jack_port_get_buffer(ports->input, AudioBufferSize));
        float* output = static_cast<float*>(jack_port_get_buffer(ports->output, AudioBufferSize));
        for (int i = 0; i < AudioBufferSize; ++i) {
            output[i] = ports->buffer[i];
        }
        return 0;
    };
    jack_set_process_callback(client, process, &ports);
    jack_activate(client);

    display(W, H, [&] (ClickEvent ev) {
        auto time = steady_clock::now();
        auto elapsed_time = time - start_time;

        auto steps_todo = duration_cast<milliseconds>(
            (elapsed_time - prev_step_time) * StepsPerSecond
        ).count() / 1000;

        // std::cout << "steps todo: " << steps_todo << std::endl;

        for (int i = 0; i < steps_todo; ++i) {
            auto t = steps / static_cast<float>(StepsPerSecond);

            if (ev.clicked) {
                impulse<<<G, B>>>(d_grid, glm::ivec2(ev.x, H - ev.y), 4.0f);
                check_kernel();
            }

            if (ev.clocked) {
                auto wall_to = glm::ivec2(ev.x, H-ev.y);
                if (wall_from != glm::ivec2{0, 0}) {
                    wall_line<<<G, B>>>(d_walls, wall_from, wall_to, 1); check_kernel();
                    check_kernel();
                }
                wall_from = wall_to;
            }

            point_source<<<G, B>>>(d_grid, glm::ivec2(100, 100), t*2.0f, max(0.0f, 1.0f - t*0.5f));
            check_kernel();

            insert_walls<<<G, B>>>(d_grid, d_walls);
            check_kernel();

            step<<<G, B>>>(d_grid_next, d_grid, d_grid_prev);
            check_kernel();

            float amp = probe(d_grid, glm::ivec2(50, 50));
            for (int i = 0; i < ProbeStepSize; ++i)
                buffer[buffer_index++] = glm::mix(prev_amp, amp, static_cast<float>(i) / ProbeStepSize);
            prev_amp = amp;

            // flush
            if (buffer_index == AudioBufferSize) {
                // TODO: send buffer to jack
                // std::cout << amp << std::endl;
                buffer_index = 0;
            }

            swap_buffers();
            ++steps;
        }

        // TODO: this might not be accurate
        prev_step_time += duration_cast<milliseconds>(
            1us * steps_todo * static_cast<int>(1000000.0f * SecondsPerStep)
        );

        draw<<<G, B>>>(d_output, d_grid, d_walls);
        check_kernel();

        check << cudaMemcpy(h_output.data(), d_output, N * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

        return h_output.data();
    });

    check << cudaFree(d_output);

    check << cudaFree(d_grid);
    check << cudaFree(d_grid_next);

    return 0;
}


            // if (ev.clicked) {
            //     impulse<<<G, B>>>(d_grid, glm::ivec2(ev.x, H - ev.y), 1.0f);
            //     point_source<<<G, B>>>(d_grid, glm::ivec2(ev.x, H-ev.y), t * 0.1f, 0.5f);
            //     check_kernel();
            // }

            // static glm::ivec2 from = glm::ivec2(0, 0);
            // if (ev.clocked) {
            //     auto to = glm::ivec2(ev.x, H-ev.y);
            //     if (from != glm::ivec2{0, 0}) {
            //         wall_line<<<G, B>>>(d_walls, from, to, 1); check_kernel();
            //         check_kernel();
            //     }
            //     from = to;
            // }
