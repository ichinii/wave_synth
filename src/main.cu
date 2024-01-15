#include <iostream>
#include <vector>
#include <chrono>

#include <glm/glm.hpp>

#include "display.h"
#include "misc.h"
#include "waves.h"
#include "audio.h"

Events input_state_to_events(const InputState& input, Events events) {
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
        events.point_source.freq = freq;
        events.point_source.pos = input.mouse_pos;
    }

    {
        bool pressed = button_pressed(input, GLFW_MOUSE_BUTTON_LEFT);
        bool drawing = events.wall.drawing || pressed;
        bool place_wall = events.wall.drawing && pressed;

        bool reset = button_pressed(input, GLFW_MOUSE_BUTTON_RIGHT);
        place_wall = !reset && place_wall;
        drawing = !reset && drawing;

        if (pressed) {
            events.wall.from = events.wall.to;
            events.wall.to = glm::vec2(input.mouse_pos);
        }

        events.wall.hover = glm::vec2(input.mouse_pos);
        events.wall.place_wall = place_wall;
        events.wall.drawing = drawing;
    }

    return events;
}

float probe(float* data, glm::ivec2 coord, cudaStream_t stream = 0) {
    float value;
    data += coord.x + coord.y * W;
    cudaMemcpyAsync(&value, data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    return value;
}

void init_walls(bool* d_walls, cudaStream_t stream) {
    int w = W-1;
    int h = H-1;
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(0, 0), glm::ivec2(w, 0), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(w, 0), glm::ivec2(w, h), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(w, h), glm::ivec2(0, h), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(0, h), glm::ivec2(0, 0), 1); check_kernel();
}

void event_source(float* d_grid, float t, const Events& events, cudaStream_t stream) {
    auto active = events.point_source.active;
    auto freq = events.point_source.freq;
    auto pos = events.point_source.pos;

    if (active) {
        point_source<<<G, B, 0, stream>>>(
            d_grid,
            pos,
            t * freq,
            0.4f
        );
        check_kernel();
    }

    point_source<<<G, B, 0, stream>>>(
        d_grid,
        glm::ivec2(100, 100),
        t*2.0f,
        max(0.0f, 1.0f - t*0.5f)
    );
    check_kernel();
}

void event_wall(bool* d_walls, const Events& events, cudaStream_t stream) {
    if (events.wall.place_wall) {
        auto from = events.wall.from;
        auto to = events.wall.to;
        wall_line<<<G, B, 0, stream>>>(d_walls, from, to, 1);
        check_kernel();
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
    check << cudaMemset(d_walls, 0, N * sizeof(bool));
    init_walls(d_walls, 0);

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

    cudaStream_t process_stream;
    cudaStream_t display_stream;
    cudaStreamCreate(&process_stream);
    cudaStreamCreate(&display_stream);
    Events events;

    auto process = [
        &,
        start_sample = 0
    ] (
        const sample_t* input,
        sample_t* output
    ) mutable -> void {
        cudaStreamSynchronize(display_stream);

        float t = static_cast<float>(start_sample) / AudioSampleRate;

        event_source(d_grid, t, events, process_stream);
        event_wall(d_walls, events, process_stream);

        insert_walls<<<G, B, 0, process_stream>>>(d_grid, d_walls);
        check_kernel();

        step<<<G, B, 0, process_stream>>>(d_grid_next, d_grid, d_grid_prev);
        check_kernel();

        cudaStreamSynchronize(process_stream);
        swap_buffers();

        start_sample += AudioBufferSize;
    };

    // run jack client on separate thread
    auto jack = create_jack_audio(process);

    // display on main thread
    display(W, H, [&] (const InputState& input) {
        cudaStreamSynchronize(process_stream);
        events = input_state_to_events(input, events);

        if (events.clear_waves) {
            check << cudaMemsetAsync(d_grid, 0, N * sizeof(float), display_stream);
            check << cudaMemsetAsync(d_grid_prev, 0, N * sizeof(float), display_stream);
        }

        if (events.clear_walls) {
            check << cudaMemsetAsync(d_walls, 0, N * sizeof(bool), display_stream);
            init_walls(d_walls, display_stream);
        }

        draw<<<G, B, 0, display_stream>>>(d_output, d_grid, d_walls, events.wall);
        check_kernel();

        check << cudaMemcpyAsync(h_output.data(), d_output, N * sizeof(glm::vec4), cudaMemcpyDeviceToHost, display_stream);

        cudaStreamSynchronize(display_stream);
        return h_output.data();
    });

    // end of program. cleanup resources
    destroy_jack_audio(jack);
    check << cudaStreamSynchronize(process_stream);
    check << cudaStreamSynchronize(display_stream);
    check << cudaStreamDestroy(process_stream);
    check << cudaStreamDestroy(display_stream);
    check << cudaFree(d_output);
    check << cudaFree(d_grid);
    check << cudaFree(d_grid_next);

    return 0;
}
