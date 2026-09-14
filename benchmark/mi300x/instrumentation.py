"""Private HIP build annotations; does not import or modify CUDA orchestration.

llama.cpp implements HIP kernels in the upstream ggml-cuda source directory.
These sources are compiled only with GGML_HIP=ON and GGML_CUDA=OFF here.
"""
import difflib
import re

OP_RANGES = r'''
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Private diagnostic annotation; no tensor values or dispatch choices change.
struct csb_operation_range {
    bool enabled;
    explicit csb_operation_range(ggml_tensor * dst, int device, bool unknown_fusion = false) {
        static const bool annotate = std::getenv("CSB_MI300X_OPS") != nullptr;
        enabled = annotate;
        if (!enabled) return;
        const ggml_tensor * a = dst->src[0];
        const ggml_tensor * b = dst->src[1];
        char label[768];
        if (unknown_fusion) {
            std::snprintf(label, sizeof(label),
                          "csb_op:role=unknown_fusion:source_role=%s:type=unknown:fusion=unclassified:geometry=unknown:phase=unknown", dst->name);
        } else if (dst->op == GGML_OP_MUL_MAT && a && b) {
            std::snprintf(label, sizeof(label), "csb_op:role=%s:type=%s:m=%lld:n=%lld:k=%lld:phase=unknown",
                          a->name, ggml_type_name(a->type), (long long) a->ne[1],
                          (long long) b->ne[1], (long long) a->ne[0]);
        } else {
            std::snprintf(label, sizeof(label), "csb_op:role=%s:type=%s:op=%s:phase=unknown",
                          dst->name, a ? ggml_type_name(a->type) : "none", ggml_op_name(dst->op));
        }
        const size_t used = std::strlen(label);
        std::snprintf(label + used, sizeof(label) - used, ":hip_device=%d:d0=%lld:d1=%lld:d2=%lld:d3=%lld",
                      device, (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3]);
        roctxRangePushA(label);
    }
    ~csb_operation_range() { if (enabled) roctxRangePop(); }
};

struct csb_fused_matrix_range {
    bool enabled;
    csb_fused_matrix_range(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * ids,
                          const ggml_cuda_mm_fusion_args_host & fusion, int device) {
        static const bool annotate = std::getenv("CSB_MI300X_OPS") != nullptr;
        enabled = annotate;
        if (!enabled) return;
        const ggml_tensor * gate = fusion.gate;
        const char * kind = gate ? "gate_up_glu" : fusion.x_scale ? "matvec_scale" : "matvec_bias";
        const bool dense_same_geometry = !ids && (!gate || ggml_are_same_shape(a, gate));
        char geometry[256];
        if (dense_same_geometry) {
            std::snprintf(geometry, sizeof(geometry), "m=%lld:n=%lld:k=%lld:geometry=per_projection",
                          (long long) a->ne[1], (long long) b->ne[1], (long long) a->ne[0]);
        } else {
            std::snprintf(geometry, sizeof(geometry), "geometry=unknown");
        }
        char label[1536];
        std::snprintf(label, sizeof(label),
                      "csb_op:role=%s%s%s:type=%s%s%s:%s:fusion=%s:up_role=%s:gate_role=%s:gate_type=%s:matrix_count=%d:bias=%d:scale=%d:glu_op=%d:phase=unknown",
                      a->name, gate ? "+" : "", gate ? gate->name : "",
                      ggml_type_name(a->type), gate ? "+" : "", gate ? ggml_type_name(gate->type) : "",
                      geometry, kind, a->name, gate ? gate->name : "none", gate ? ggml_type_name(gate->type) : "none", gate ? 2 : 1,
                      int(fusion.x_bias != nullptr || fusion.gate_bias != nullptr),
                      int(fusion.x_scale != nullptr || fusion.gate_scale != nullptr),
                      gate ? int(fusion.glu_op) : -1);
        const size_t used = std::strlen(label);
        std::snprintf(label + used, sizeof(label) - used, ":hip_device=%d", device);
        roctxRangePushA(label);
    }
    ~csb_fused_matrix_range() { if (enabled) roctxRangePop(); }
};
'''

SERVER_OLD = r'''        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });'''

SERVER_NEW = r'''        queue_tasks.yield_to_queue([&]() {
            static const bool csb_annotate = std::getenv("CSB_MI300X_OPS") != nullptr;
            bool csb_selected = false;
            if (csb_annotate) {
                int csb_prefill = 0;
                for (int i = off; i < off + batch_view.n_tokens; ++i) {
                    csb_prefill += batch.tokens[i].is_prompt;
                }
                const int csb_decode = batch_view.n_tokens - csb_prefill;
                const char * csb_phase = csb_prefill == 0 ? "decode" : csb_decode == 0 ? "prefill" : "mixed";
                if (csb_mi300x_batch_begin) csb_selected = csb_mi300x_batch_begin(csb_phase, batch_view.n_tokens);
                char csb_label[256];
                std::snprintf(csb_label, sizeof(csb_label),
                              "csb_batch:phase=%s:prefill_tokens=%d:decode_tokens=%d:width=%d",
                              csb_phase, csb_prefill, csb_decode, batch_view.n_tokens);
                roctxRangePushA(csb_label);
            }
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
            if (csb_annotate) roctxRangePop();
            if (csb_selected && csb_mi300x_batch_end) csb_mi300x_batch_end();
        });'''

def annotate_operator(original):
    signature = "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {"
    loop = "                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);"
    fuse = "static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {"
    if 'CSB_MI300X_OPS' in original or any(original.count(s) != 1 for s in (signature, loop, fuse)):
        raise RuntimeError("Pinned HIP annotation insertion points changed")
    updated = original.replace(signature, OP_RANGES + "\n" + signature)
    updated = updated.replace(loop, "                csb_operation_range csb_range(node, cuda_ctx->device);\n" + loop)
    updated = updated.replace(fuse, fuse + "\n    csb_operation_range csb_unknown_fusion(cgraph->nodes[i], cuda_ctx->device, true);")
    pattern = r"^( +)(ggml_cuda_mul_mat_vec_[fq]\(\*cuda_ctx, src0, src1, ids, [^\n]+, &fusion_data\);)$"
    updated, count = re.subn(pattern, r"\1csb_fused_matrix_range csb_fused_range(src0, src1, ids, fusion_data, cuda_ctx->device);\n\1\2", updated, flags=re.MULTILINE)
    if count != 9:
        raise RuntimeError(f"Expected nine fused HIP matvec sites, received {count}")
    return updated


def annotate_server(original):
    if original.count(SERVER_OLD) != 1:
        raise RuntimeError("Pinned server batch annotation insertion point changed")
    declarations = ('extern "C" int csb_mi300x_batch_begin(const char *, int) __attribute__((weak));\n'
                    'extern "C" void csb_mi300x_batch_end() __attribute__((weak));\n')
    return '#include <rocprofiler-sdk-roctx/roctx.h>\n#include <cstdio>\n#include <cstdlib>\n' + declarations + original.replace(SERVER_OLD, SERVER_NEW)


def instrument(source, rocm):
    differences = []
    for relative, transform in [("ggml/src/ggml-cuda/ggml-cuda.cu", annotate_operator),
                                ("tools/server/server-context.cpp", annotate_server)]:
        path = source / relative
        original = path.read_text()
        updated = transform(original)
        path.write_text(updated)
        differences.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + relative, tofile="b/" + relative))
    # Both targets compile code that calls ROCTx (server-context is a static lib).
    for relative, target in [("ggml/src/ggml-hip/CMakeLists.txt", "ggml-hip"),
                             ("tools/server/CMakeLists.txt", "server-context")]:
        path = source / relative
        original = path.read_text()
        updated = original + f'\ntarget_include_directories({target} PRIVATE "{rocm}/include")\ntarget_link_libraries({target} PUBLIC "{rocm}/lib/librocprofiler-sdk-roctx.so")\n'
        path.write_text(updated)
        differences.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                               fromfile="a/" + relative, tofile="b/" + relative))
    return "".join(differences)
