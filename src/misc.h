#pragma once

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
