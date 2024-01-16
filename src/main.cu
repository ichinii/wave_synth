#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <signalsmith-stretch.h>

#include "display.h"
#include "misc.h"
#include "waves.h"
#include "audio.h"
#include "draw.h"

void pitch_shift(float* dst, float* src, float factor) {
    signalsmith::stretch::SignalsmithStretch<float> stretch;

    int channels = 1;
    stretch.presetDefault(channels, AudioSampleRate);
    stretch.setTransposeFactor(16);

    float** input = &src;
    float** output = &dst;
    stretch.process(input, AudioBufferSize, output, AudioBufferSize);
}

void init_walls(bool* d_walls, cudaStream_t stream) {
    int w = W-1;
    int h = H-1;
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(0, 0), glm::ivec2(w, 0), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(w, 0), glm::ivec2(w, h), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(w, h), glm::ivec2(0, h), 1); check_kernel();
    wall_line<<<G, B, 0, stream>>>(d_walls, glm::ivec2(0, h), glm::ivec2(0, 0), 1); check_kernel();
}

void process_point_source_event(float* d_grid, float t, Events::PointSource ev, cudaStream_t stream) {
    if (ev.active) {
        point_source<<<G, B, 0, stream>>>(
            d_grid,
            ev.pos,
            t * ev.freq,
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

void process_wall_event(bool* d_walls, Events::Wall ev, cudaStream_t stream) {
    if (ev.place) {
        auto from = ev.from;
        auto to = ev.to;
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

    cudaStream_t process_stream;
    cudaStream_t display_stream;
    cudaStreamCreate(&process_stream);
    cudaStreamCreate(&display_stream);
    Events events;

    auto buffer = std::vector<float>(0, AudioBufferSize);

    auto process = [
        &,
        start_sample = 0
    ] (
        const sample_t* input,
        sample_t* output
    ) mutable -> void {
        cudaStreamSynchronize(display_stream);

        float amps[ProcessedSamplesPerAudioBuffer];

        for (int i = 0; i < ProcessedSamplesPerAudioBuffer; ++i) {
            float t = static_cast<float>(start_sample) / AudioSampleRate;
            process_point_source_event(d_grid, t, events.point_source, process_stream);
            process_wall_event(d_walls, events.wall, process_stream);

            insert_walls<<<G, B, 0, process_stream>>>(d_grid, d_walls);
            check_kernel();

            step<<<G, B, 0, process_stream>>>(d_grid_next, d_grid, d_grid_prev);
            cycle_swap(d_grid_next, d_grid, d_grid_prev);

            amps[i] = sink(d_grid, glm::ivec2(), process_stream);

            start_sample += AudioBufferSize / ProcessedSamplesPerAudioBuffer;
        }

        if (output) {
            buffer.insert(buffer.end(), amps, amps + ProcessedSamplesPerAudioBuffer);
            pitch_shift(output, &buffer[buffer.size() - AudioBufferSize], 1.0f);
        }

        cudaStreamSynchronize(process_stream);
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

            // TODO: remove this when we have propper thread sync
            check << cudaMemsetAsync(d_grid_next, 0, N * sizeof(float), display_stream);
        }

        if (events.clear_walls) {
            check << cudaMemsetAsync(d_walls, 0, N * sizeof(bool), display_stream);
            init_walls(d_walls, display_stream);
        }

        if (!jack) {
            process(nullptr, nullptr);
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
