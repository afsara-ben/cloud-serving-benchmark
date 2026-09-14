#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstdio>
#include <cstdlib>
static void check(hipError_t s) {
    if (s != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(s)); std::exit(2); }
}
__global__ void csb_mi300x_probe(float *p) { p[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x + 1.0f; }
int main() {
    float *p = nullptr;
    check(hipMalloc(&p, 16 * 256 * sizeof(float)));
    csb_mi300x_probe<<<1, 256>>>(p); check(hipDeviceSynchronize());
    if (roctxProfilerResume(0)) return 3;
    csb_mi300x_probe<<<16, 256>>>(p); check(hipDeviceSynchronize());
    if (roctxProfilerPause(0)) return 4;
    csb_mi300x_probe<<<1, 256>>>(p); check(hipDeviceSynchronize());
    check(hipFree(p));
}
