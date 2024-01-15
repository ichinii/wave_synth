#pragma once

#include <any>
#include "misc.h"
#include <jack/jack.h>
#include <fftw3.h>

using sample_t = jack_default_audio_sample_t;
static_assert(std::is_same_v<float, sample_t>);
// TODO: use sample_t in process

struct JackData {
    jack_client_t* client;
    jack_port_t* input;
    jack_port_t* output;
    void* process_fn;
};

void print_jack_status(jack_status_t status) {
    std::cout << "jack status: " << (status ? "error" : "ok") << std::endl;
}

template <typename ProcessFn>
JackData* create_jack_audio(ProcessFn& process_fn) {
    jack_status_t status;
    auto client = jack_client_open("wavy_synth", JackNoStartServer, &status);
    print_jack_status(status);

    if (status)
        return nullptr;

    auto input = jack_port_register(
        client,
        "input",
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortIsInput,
        AudioBufferSize
    );
    assert(input);

    auto output = jack_port_register(
        client,
        "output",
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortIsOutput,
        AudioBufferSize
    );
    assert(output);

    auto process_wrapper = [] (jack_nframes_t nframes, void* args) -> int {
        assert(nframes == AudioBufferSize);
        auto self = static_cast<JackData*>(args);
        const sample_t* input = static_cast<sample_t*>(jack_port_get_buffer(self->input, AudioBufferSize));
        sample_t* output = static_cast<sample_t*>(jack_port_get_buffer(self->output, AudioBufferSize));
        auto& process_fn = *static_cast<ProcessFn*>(self->process_fn);
        process_fn(input, output);
        return 0;
    };

    auto jack = new JackData{client, input, output, static_cast<void*>(&process_fn)};
    assert(!jack_set_process_callback(client, process_wrapper, jack));
    assert(!jack_activate(client));
    return jack;
}

void destroy_jack_audio(JackData* jack) {
    if (jack) {
        jack_deactivate(jack->client);
        jack_port_unregister(jack->client, jack->input);
        jack_port_unregister(jack->client, jack->output);
        jack_client_close(jack->client);
        delete jack;
    }
}
