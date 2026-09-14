// Used ONLY by the annotated HIP server, through rocprofv3 --preload.
#include <hip/hip_runtime_api.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <string>
#include <cstring>
#include <unistd.h>

// The server calls these hooks around an actual llama_decode batch. A bounded
// phase capture records ALL dispatches in that batch, including the second GPU.
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool armed = false;
static int batches = 0;
static thread_local bool selected_batch = false;

static void mark(const std::string &path, const char *message) {
    const std::string temporary = path + ".tmp";
    if (FILE *f = std::fopen(temporary.c_str(), "w")) {
        std::fputs(message, f); std::fclose(f); std::rename(temporary.c_str(), path.c_str());
    }
}
static hipError_t drain() {
    int count = 0, previous = 0;
    hipError_t saved = hipGetDevice(&previous);
    if (saved != hipSuccess) return saved;
    hipError_t s = hipGetDeviceCount(&count);
    for (int d = 0; s == hipSuccess && d < count; ++d) {
        s = hipSetDevice(d);
        if (s == hipSuccess) s = hipDeviceSynchronize();
    }
    const hipError_t restore = hipSetDevice(previous);
    return s == hipSuccess ? restore : s;
}
extern "C" int csb_mi300x_batch_begin(const char *phase, int width) {
    const char *wanted = std::getenv("CSB_MI300X_PHASE");
    if (!wanted) return 0;
    pthread_mutex_lock(&gate_mutex);
    const int limit = std::atoi(std::getenv("CSB_MI300X_BATCHES"));
    const int target_width = std::atoi(std::getenv("CSB_MI300X_WIDTH"));
    if (!armed || batches >= limit || std::strcmp(phase, wanted) ||
        (target_width && width != target_width)) {
        pthread_mutex_unlock(&gate_mutex);
        return 0;
    }
    const std::string prefix(std::getenv("CSB_MI300X_GATE"));
    hipError_t s = drain();
    if (s != hipSuccess || roctxProfilerResume(0)) {
        mark(prefix + ".error", "Phase gate resume/drain failed");
        pthread_mutex_unlock(&gate_mutex);
        return 0;
    }
    selected_batch = true;
    // Hold the mutex until end, so the watcher cannot stop an in-flight batch.
    return 1;
}
extern "C" void csb_mi300x_batch_end() {
    if (!selected_batch) return;
    const std::string prefix(std::getenv("CSB_MI300X_GATE"));
    hipError_t s = drain();
    if (s != hipSuccess || roctxProfilerPause(0))
        mark(prefix + ".error", "Phase gate pause/drain failed");
    ++batches;
    mark(prefix + ".batches", std::to_string(batches).c_str());
    selected_batch = false;
    pthread_mutex_unlock(&gate_mutex);
}
static void *watch(void *arg) {
    const std::string prefix(static_cast<char *>(arg));
    std::free(arg);
    while (access((prefix + ".start").c_str(), F_OK)) usleep(10000);
    const bool phase_mode = std::getenv("CSB_MI300X_PHASE") != nullptr;
    hipError_t s = drain();
    if (s != hipSuccess || (!phase_mode && roctxProfilerResume(0))) {
        mark(prefix + ".error", s != hipSuccess ? hipGetErrorString(s) : "ROCTx resume failed");
        return nullptr;
    }
    pthread_mutex_lock(&gate_mutex);
    armed = true;
    pthread_mutex_unlock(&gate_mutex);
    mark(prefix + ".started", "ok");
    while (access((prefix + ".stop").c_str(), F_OK)) usleep(10000);
    pthread_mutex_lock(&gate_mutex);
    armed = false;
    s = drain();
    if (s != hipSuccess || (!phase_mode && roctxProfilerPause(0)))
        mark(prefix + ".error", s != hipSuccess ? hipGetErrorString(s) : "ROCTx pause failed");
    else mark(prefix + ".stopped", "ok");
    pthread_mutex_unlock(&gate_mutex);
    return nullptr;
}
__attribute__((constructor)) static void initialize() {
    const char *prefix = std::getenv("CSB_MI300X_GATE");
    if (!prefix || !*prefix) return;
    // --selected-regions starts paused. Avoid invoking HIP during loader initialization.
    pthread_t thread;
    char *argument = static_cast<char *>(std::malloc(std::string(prefix).size() + 1));
    if (!argument) { mark(std::string(prefix) + ".error", "malloc failed"); return; }
    std::string(prefix).copy(argument, std::string(prefix).size());
    argument[std::string(prefix).size()] = 0;
    if (pthread_create(&thread, nullptr, watch, argument)) {
        std::free(argument); mark(std::string(prefix) + ".error", "pthread_create failed"); return;
    }
    pthread_detach(thread);
    mark(std::string(prefix) + ".ready", std::to_string(getpid()).c_str());
}
