#pragma once

#include "common.h"

constexpr int W = 512;
constexpr int H = 512;
constexpr int N = W*H;

constexpr int B = 128;
constexpr int G = N/B;

#define pi 3.14159265358979323844f
#define pi2 (pi*2.0f)

#if defined(debug_prints)
#define __assert(val, str) if (!val) printf( \
    "assertion at line (%d), gid (%d): %s\n", \
    __LINE__, threadIdx.x + blockIdx.x*blockDim.x, str);
#else
#define __assert(val, str)
#endif

struct ErrorPrinter {
    int line;

    void operator<< (cudaError_t err) {
        auto last = cudaGetLastError();
        if (last != cudaSuccess && last != err) {
            std::cout
                << "catched lost cuda error above line (" << line << "): "
                << cudaGetErrorString(err) << std::endl;

        }
        if (err != cudaSuccess)
            std::cout
                << "cuda error at line (" << line << "): "
                << cudaGetErrorString(err) << std::endl;
    }
};
#define check ErrorPrinter{__LINE__}
#define check_kernel ErrorPrinter{__LINE__} << cudaGetLastError
