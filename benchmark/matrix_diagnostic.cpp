// Standalone complete-operation probe using the same public ggml APIs as test-backend-ops.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static float value(uint64_t index, uint64_t seed) {
    return (float(mix(index ^ seed) >> 40) / 8388608.0f - 1.0f) * 0.1f;
}

static ggml_type parse_type(std::string name) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto * traits = ggml_get_type_traits(ggml_type(i));
        if (traits && traits->type_name && name == traits->type_name) return ggml_type(i);
    }
    throw std::runtime_error("Unknown ggml type: " + name + "; use lowercase, e.g. q2_K or iq1_m");
}

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) try {
    int64_t m = 28672, n = 8, k = 8192;
    int iterations = 20, warmup = 3, samples = 32, threads = 4;
    double warmup_ms = 250.0;
    uint64_t seed = 20260912;
    std::string type_name = "q2_K", backend_name = "CUDA0", library_dir, output, weights_cache, role = "ffn_gate_up", phase = "decode";
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::cout << "matrix-diagnostic --library-dir DIR --backend CUDA0|CPU --type q2_K|q3_K|q4_K|q5_K|q6_K|iq1_m|f16 "
                         "--m ROWS --n WIDTH --k INNER --iterations 20 --warmup 3 --warmup-ms 250 --samples 32 --seed 20260912 "
                         "--role NAME --phase decode|prefill|unknown --threads 4 --weights-cache FILE --output FILE\n";
            return 0;
        }
        require(i + 1 < argc, "Missing value for " + flag);
        const std::string arg = argv[++i];
        if (flag == "--m") m = std::stoll(arg);
        else if (flag == "--n") n = std::stoll(arg);
        else if (flag == "--k") k = std::stoll(arg);
        else if (flag == "--iterations") iterations = std::stoi(arg);
        else if (flag == "--warmup") warmup = std::stoi(arg);
        else if (flag == "--warmup-ms") warmup_ms = std::stod(arg);
        else if (flag == "--samples") samples = std::stoi(arg);
        else if (flag == "--threads") threads = std::stoi(arg);
        else if (flag == "--seed") seed = std::stoull(arg);
        else if (flag == "--type") type_name = arg;
        else if (flag == "--backend") backend_name = arg;
        else if (flag == "--library-dir") library_dir = arg;
        else if (flag == "--output") output = arg;
        else if (flag == "--weights-cache") weights_cache = arg;
        else if (flag == "--role") role = arg;
        else if (flag == "--phase") phase = arg;
        else throw std::runtime_error("Unknown argument " + flag);
    }
    require(m > 0 && n > 0 && k > 0 && iterations > 0 && warmup > 0 && samples >= 2 && threads > 0, "Invalid dimensions or repetition counts");
    require(std::isfinite(warmup_ms) && warmup_ms >= 0, "Warmup duration must be finite and nonnegative");
    require(phase == "prefill" || phase == "decode" || phase == "unknown", "Invalid phase");
    require(role.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos,
            "Role must be a simple tensor identifier");
    const ggml_type type = parse_type(type_name);
    require(k % ggml_blck_size(type) == 0, "K must be a multiple of the quantization block size");
    require(!library_dir.empty(), "An explicit --library-dir is required to pin the backend");
    const std::string lib = library_dir + (backend_name == "CPU" ? "/libggml-cpu.so" : "/libggml-cuda.so");
    // Non-plugin builds register their linked backends automatically.
    if (!ggml_backend_dev_by_name(backend_name.c_str())) {
        require(ggml_backend_load(lib.c_str()) != nullptr, "Could not load " + lib);
    }
    ggml_backend_t backend = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
    require(backend != nullptr, "Requested backend is unavailable; no CPU fallback is allowed");
    const bool is_gpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    require(is_gpu || backend_name == "CPU", "Unexpected backend device type");

    // Separate weights match the serving backend's weight-buffer treatment.
    const ggml_init_params params = {ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true};
    ggml_context * weights_context = ggml_init(params);
    ggml_context * context = ggml_init(params);
    require(context && weights_context, "Context allocation failed");
    ggml_tensor * weights = ggml_new_tensor_2d(weights_context, type, k, m);
    ggml_set_name(weights, role.c_str());
    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, n);
    ggml_set_name(input, "activation");
    ggml_tensor * result = ggml_mul_mat(context, weights, input);
    ggml_set_name(result, (role + "_result").c_str());
    require(ggml_backend_supports_op(backend, result), "Backend does not support operation");
    ggml_cgraph * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context, backend);
    ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(weights_buffer && compute_buffer, "Device allocation failed");
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t row_bytes = ggml_row_size(type, k);
    std::vector<uint8_t> quantized(row_bytes * m);
    const uint64_t cache_header[] = {0x4353424d41543031ULL, uint64_t(type), uint64_t(m), uint64_t(k), seed};
    bool cache_hit = false;
    if (!weights_cache.empty()) {
        std::ifstream cache(weights_cache, std::ios::binary);
        if (cache) {
            uint64_t observed_header[5] = {};
            cache.read(reinterpret_cast<char *>(observed_header), sizeof(observed_header));
            require(std::memcmp(cache_header, observed_header, sizeof(cache_header)) == 0, "Weight cache identity mismatch");
            cache.read(reinterpret_cast<char *>(quantized.data()), quantized.size());
            require(bool(cache) && cache.peek() == EOF, "Weight cache length mismatch");
            cache_hit = true;
        }
    }
    if (!cache_hit) {
        ggml_quantize_init(type);
        std::vector<std::thread> workers;
        for (int worker = 0; worker < threads; ++worker) workers.emplace_back([&, worker]() {
            std::vector<float> weight_chunk(size_t(k) * 64), importance(k, 1.0f);
            for (int64_t row = worker * 64; row < m; row += threads * 64) {
                const int64_t rows = std::min<int64_t>(64, m - row);
                for (int64_t i = 0; i < rows * k; ++i) weight_chunk[i] = value(uint64_t(row * k + i), seed);
                ggml_quantize_chunk(type, weight_chunk.data(), quantized.data() + row_bytes * row,
                                    0, rows, k, ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
            }
        });
        for (auto & worker : workers) worker.join();
        if (!weights_cache.empty()) {
            const std::string temporary = weights_cache + ".partial";
            std::ofstream cache(temporary, std::ios::binary);
            cache.write(reinterpret_cast<const char *>(cache_header), sizeof(cache_header));
            cache.write(reinterpret_cast<const char *>(quantized.data()), quantized.size());
            cache.close();
            require(bool(cache), "Failed to write weight cache");
            require(std::rename(temporary.c_str(), weights_cache.c_str()) == 0, "Failed to publish weight cache");
        }
    }
    std::vector<float> activations(k * n), actual(m * n);
    for (int64_t i = 0; i < k * n; ++i) activations[i] = value(i, seed ^ 0xd1b54a32d192ed03ULL);
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));
    const auto compute = [&]() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Graph execution failed");
        ggml_backend_synchronize(backend);
    };
    compute();
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));

    // Compare sampled outputs against the actual quantized weights, not the source floats.
    std::vector<float> restored(k);
    const auto * traits = ggml_get_type_traits(type);
    require(traits->to_float != nullptr, "Reference dequantizer unavailable");
    const int sample_rows = int(std::min<int64_t>(samples, m));
    double error_squared = 0, reference_squared = 0, max_absolute = 0;
    bool finite = true;
    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = int64_t(sample) * (m - 1) / std::max(1, sample_rows - 1);
        traits->to_float(quantized.data() + row_bytes * row, restored.data(), k);
        for (int64_t col = 0; col < n; ++col) {
            double expected = 0;
            for (int64_t inner = 0; inner < k; ++inner) expected += double(restored[inner]) * activations[col * k + inner];
            const double observed = actual[col * m + row];
            const double error = observed - expected;
            finite = finite && std::isfinite(observed);
            error_squared += error * error;
            reference_squared += expected * expected;
            max_absolute = std::max(max_absolute, std::abs(error));
        }
    }
    const double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
    const double reference_rms = std::sqrt(reference_squared / (sample_rows * n));
    const double maximum_normalized_error = max_absolute / std::max(reference_rms, 1e-30);
    const double tolerance = type == GGML_TYPE_F16 ? 0.003 : 0.03;
    const bool correct = finite && relative_l2 <= tolerance && maximum_normalized_error <= 0.15;

    // The API gate excludes allocation, quantization, warmup, and correctness from capture.
    using profiler_function = int (*)();
    profiler_function start = nullptr, stop = nullptr;
    void * cudart = nullptr;
    if (is_gpu) {
        for (const char * name : {"libcudart.so.13", "libcudart.so.12", "libcudart.so"}) {
            cudart = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (cudart) break;
        }
        require(cudart, "CUDA profiler API is unavailable");
        start = reinterpret_cast<profiler_function>(dlsym(cudart, "cudaProfilerStart"));
        stop = reinterpret_cast<profiler_function>(dlsym(cudart, "cudaProfilerStop"));
        require(start && stop, "CUDA profiler API symbols unavailable");
    }
    std::vector<double> durations;
    durations.reserve(iterations);
    const std::string label = "csb_op:role=" + role + ":type=" + type_name + ":m=" + std::to_string(m) +
                              ":n=" + std::to_string(n) + ":k=" + std::to_string(k) + ":phase=" + phase;
    int warmup_actual_iterations = 0;
    double warmup_actual_ms = 0;
    if (correct) {
        // Restore sustained work after the CPU reference calculation, outside capture.
        const auto warmup_start = std::chrono::steady_clock::now();
        do {
            compute();
            ++warmup_actual_iterations;
            warmup_actual_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warmup_start).count();
        } while (warmup_actual_iterations < warmup || warmup_actual_ms < warmup_ms);
        if (start) require(start() == 0, "cudaProfilerStart failed");
        for (int i = 0; i < iterations; ++i) {
            nvtxRangePushA(label.c_str());
            const auto before = std::chrono::steady_clock::now();
            compute();
            const auto after = std::chrono::steady_clock::now();
            nvtxRangePop();
            durations.push_back(std::chrono::duration<double, std::micro>(after - before).count());
        }
        if (stop) require(stop() == 0, "cudaProfilerStop failed");
    }
    std::ofstream file;
    if (!output.empty()) { file.open(output); require(bool(file), "Cannot open output"); }
    std::ostream & out = output.empty() ? std::cout : file;
    out << std::setprecision(10) << "{\n  \"schema\": 2, \"backend\": \"" << backend_name << "\", \"type\": \"" << type_name
        << "\", \"role\": \"" << role << "\", \"phase\": \"" << phase << "\",\n  \"m\": " << m << ", \"n\": " << n << ", \"k\": " << k
        << ", \"seed\": " << seed << ", \"weight_bytes\": " << quantized.size() << ", \"warmup\": " << warmup
        << ", \"warmup_ms\": " << warmup_ms << ", \"warmup_actual_iterations\": " << warmup_actual_iterations
        << ", \"warmup_actual_ms\": " << warmup_actual_ms
        << ",\n  \"warmup_position\": \"after_correctness_before_capture\", \"correctness_operations\": 1"
        << ",\n  \"input_provenance\": \"Deterministic synthetic uniform weights and activations; exact model matrix geometry; unit importance values when required\""
        << ",\n  \"correctness\": {\"passed\": " << (correct ? "true" : "false") << ", \"sampled_outputs\": " << sample_rows * n
        << ", \"relative_l2\": " << (std::isfinite(relative_l2) ? std::to_string(relative_l2) : "null")
        << ", \"relative_l2_tolerance\": " << tolerance << ", \"max_error_over_reference_rms\": "
        << (std::isfinite(maximum_normalized_error) ? std::to_string(maximum_normalized_error) : "null")
        << ", \"maximum_normalized_tolerance\": 0.15, \"reference\": \"CPU float64 dot product of dequantized weights and original FP32 activations\"},"
        << "\n  \"measurement\": \"Synchronized complete ggml operation, including activation conversion, dequantization and auxiliary kernels; repeated weights are warm\","
        << "\n  \"complete_operation_us\": [";
    for (size_t i = 0; i < durations.size(); ++i) out << (i ? ", " : "") << durations[i];
    out << "]\n}\n";
    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(context);
    ggml_free(weights_context);
    ggml_backend_free(backend);
    ggml_quantize_free();
    if (cudart) dlclose(cudart);
    return correct ? 0 : 3;
} catch (const std::exception & error) {
    std::cerr << "matrix-diagnostic: " << error.what() << '\n';
    return 2;
}
