#pragma once

constexpr int AudioSampleRate = 44100;
constexpr int AudioBufferSize = 512;
constexpr int ProcessedSamplesPerAudioBuffer = 16;
// static_assert(AudioSampleRate % AudioBufferSize == 0);
