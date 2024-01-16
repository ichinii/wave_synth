#pragma once

__global__
void draw(glm::vec4* output, float* data, bool* walls, Events::Wall wall_event) {
    int gid = threadIdx.x + blockIdx.x*blockDim.x;

    // draw waves
    float value = data[gid];
    float high = max(0.0f, value);
    float low = max(0.0f, -value);
    float overflow = max(0.0f, abs(value) - 1.0f);
    glm::vec3 c = glm::vec3(high, low, overflow);

    // draw wall
    if (walls[gid])
        c = glm::vec3(1);

    // draw wall hover indicator
    if (wall_event.drawing) {
        auto coord = index_to_coord(gid);
        bool inside = sd_segment(coord, wall_event.to, wall_event.hover, 7);
        if (inside)
            c += 0.01f;
    }

    // gamma
    c = glm::pow(glm::vec3(c), glm::vec3(1.0f/2.2f));

    output[gid] = glm::vec4(c, 1.0f);
}
