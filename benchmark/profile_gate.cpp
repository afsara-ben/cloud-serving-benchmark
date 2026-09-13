// A process-local GPU profiler gate. The HTTP driver creates .start/.stop;
// acknowledgements ensure warmup and model loading remain outside the capture.
#ifdef CSB_USE_ROCM
#include <hip/hip_runtime_api.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#else
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <string>
#include <unistd.h>

static void write_file(const std::string &path, const char *value) {
    if (FILE *file = fopen(path.c_str(), "w")) {
        fputs(value, file);
        fclose(file);
    }
}

static void *watch_gate(void *argument) {
    const std::string prefix(static_cast<char *>(argument));
    free(argument);
    while (access((prefix + ".start").c_str(), F_OK) != 0) usleep(10000);
#ifdef CSB_USE_ROCM
    hipError_t status = hipSetDevice(0);  // HIP_VISIBLE_DEVICES selects the GPU.
    if (status != hipSuccess || roctxProfilerResume(0) != 0) {
        write_file(prefix + ".error", status != hipSuccess ? hipGetErrorString(status) : "ROCTx resume failed");
        return nullptr;
    }
    write_file(prefix + ".started", "ok\n");
    while (access((prefix + ".stop").c_str(), F_OK) != 0) usleep(10000);
    status = hipDeviceSynchronize();
    const int paused = roctxProfilerPause(0);
    write_file(prefix + (status == hipSuccess && paused == 0 ? ".stopped" : ".error"),
               status != hipSuccess ? hipGetErrorString(status) : paused == 0 ? "ok\n" : "ROCTx pause failed");
#else
    // The server uses the process-shared primary contexts on CUDA_VISIBLE_DEVICES.
    // Nsight Systems gates the session on the first Start/Stop pair; NCU uses
    // the Runtime API's current-context scope and needs each device enabled.
    const char *tool = getenv("CSB_PROFILE_TOOL");
    const bool each_context = tool && strcmp(tool, "ncu") == 0;
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    std::string devices = "{\"visible_device_count\":" + std::to_string(device_count)
        + ",\"peer_access_capability\":[";
    bool first_pair = true;
    for (int device = 0; status == cudaSuccess && device < device_count; ++device) {
        status = cudaSetDevice(device);
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        for (int peer = 0; status == cudaSuccess && peer < device_count; ++peer) {
            if (peer == device) continue;
            int capable = 0;
            status = cudaDeviceCanAccessPeer(&capable, device, peer);
            if (!first_pair) devices += ",";
            first_pair = false;
            devices += "{\"source\":" + std::to_string(device) + ",\"destination\":"
                + std::to_string(peer) + ",\"capable\":" + (capable ? "true" : "false") + "}";
        }
    }
    devices += "]}\n";
    const int profiler_contexts = each_context ? device_count : 1;
    for (int device = 0; status == cudaSuccess && device < profiler_contexts; ++device) {
        status = cudaSetDevice(device);
        if (status == cudaSuccess) status = cudaProfilerStart();
    }
    if (status != cudaSuccess) {
        write_file(prefix + ".error", cudaGetErrorString(status));
        return nullptr;
    }
    write_file(prefix + ".devices", devices.c_str());
    write_file(prefix + ".started", "ok\n");
    while (access((prefix + ".stop").c_str(), F_OK) != 0) usleep(10000);
    // Drain every device BEFORE the first Stop, which closes the nsys capture.
    for (int device = 0; status == cudaSuccess && device < device_count; ++device) {
        status = cudaSetDevice(device);
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
    }
    for (int device = 0; status == cudaSuccess && device < profiler_contexts; ++device) {
        status = cudaSetDevice(device);
        if (status == cudaSuccess) status = cudaProfilerStop();
    }
    write_file(prefix + (status == cudaSuccess ? ".stopped" : ".error"),
               status == cudaSuccess ? "ok\n" : cudaGetErrorString(status));
#endif
    return nullptr;
}

__attribute__((constructor)) static void initialize_gate() {
    const char *prefix = getenv("CSB_PROFILE_PREFIX");
    if (!prefix || !*prefix) return;
    char pid[32];
    snprintf(pid, sizeof(pid), "%ld\n", static_cast<long>(getpid()));
    write_file(std::string(prefix) + ".pid", pid);
    pthread_t thread;
    char *copy = strdup(prefix);
    if (pthread_create(&thread, nullptr, watch_gate, copy) == 0) {
        pthread_detach(thread);
    } else {
        free(copy);
        write_file(std::string(prefix) + ".error", "pthread_create failed\n");
    }
}
