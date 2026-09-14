#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(2);
    }
}
static std::string escaped(const char *s) {
    std::string out;
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') out += '\\';
        if (static_cast<unsigned char>(*s) >= 32) out += *s;
    }
    return out;
}
int main() {
    int count = 0, runtime = 0, driver = 0;
    check(hipGetDeviceCount(&count));
    check(hipRuntimeGetVersion(&runtime));
    check(hipDriverGetVersion(&driver));
    std::printf("{\"runtime_version\":%d,\"driver_version\":%d,\"devices\":[", runtime, driver);
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        char bus[64]{};
        size_t free_bytes = 0, total_bytes = 0;
        check(hipSetDevice(i));
        check(hipGetDeviceProperties(&p, i));
        check(hipDeviceGetPCIBusId(bus, sizeof(bus), i));
        check(hipMemGetInfo(&free_bytes, &total_bytes));
        std::printf("%s{\"index\":%d,\"name\":\"%s\",\"arch\":\"%s\",\"pci_bus\":\"%s\","
                    "\"total_bytes\":%zu,\"free_bytes\":%zu,\"compute_units\":%d,\"wave_size\":%d}",
                    i ? "," : "", i, escaped(p.name).c_str(), escaped(p.gcnArchName).c_str(),
                    escaped(bus).c_str(), total_bytes, free_bytes, p.multiProcessorCount, p.warpSize);
    }
    std::puts("]}");
}
