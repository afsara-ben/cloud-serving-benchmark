"""gfx942 metric definitions. All reductions happen in the profiler, per dispatch.

Sources and formulas are exported with every capture. In particular, MFMA is a
subset of VALU; adding those instruction counters would double count work.
"""
from .common import write

SOURCES = {
    'counter_units': 'https://rocm.docs.amd.com/en/docs-6.3.1/conceptual/gpu-arch/mi300-mi200-performance-counters.html',
    'pipeline': 'https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.0/conceptual/pipeline-metrics.html',
    'sdk_definitions': 'https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/share/rocprofiler-sdk/counter_defs.yaml',
    'gfx942_pipeline': 'https://github.com/ROCm/rocprofiler-compute/blob/develop_deprecated/src/rocprof_compute_soc/analysis_configs/gfx942/1100_compute_units_compute_pipeline.yaml',
    'registers': 'https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html',
}

GROUPS = {
    'instructions': ['SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_SALU', 'SQ_INSTS_MFMA'],
    'memory': ['FETCH_SIZE', 'WRITE_SIZE'],
    'cache': ['TCC_HIT', 'TCC_MISS'],
    'occupancy': ['MeanOccupancyPerActiveCU'],
    'scalar_fp32': ['SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32', 'SQ_INSTS_VALU_FMA_F32', 'FETCH_SIZE', 'WRITE_SIZE'],
    'mfma_f16': ['SQ_INSTS_VALU_MFMA_MOPS_F16', 'FETCH_SIZE', 'WRITE_SIZE'],
    'mfma_i8': ['SQ_INSTS_VALU_MFMA_MOPS_I8', 'FETCH_SIZE', 'WRITE_SIZE'],
    # Keep the operands needed for each ratio in one hardware pass.
    'attribution': ['SQ_INSTS', 'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_CVT'],
    'ipc': ['SQ_INSTS', 'SQ_BUSY_CU_CYCLES'],
    'activity': ['VALU_BUSY_PERCENT', 'MFMA_BUSY_PERCENT'],
    'scratch': ['TA_BUFFER_READ_WAVEFRONTS_sum', 'TA_BUFFER_WRITE_WAVEFRONTS_sum'],
}
BOTTLE_GROUPS = ('memory', 'attribution', 'ipc', 'activity', 'occupancy', 'scratch', 'mfma_i8', 'scalar_fp32', 'mfma_f16')

RAW = {n for names in GROUPS.values() for n in names if n.startswith('SQ_') or n in ('TCC_HIT', 'TCC_MISS')}
EXPRESSIONS = {n: f'reduce({n},sum)' for n in RAW}
EXPRESSIONS.update({
    'VALU_BUSY_PERCENT': '100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)',
    'MFMA_BUSY_PERCENT': '100*reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/(4*CU_NUM*reduce(GRBM_GUI_ACTIVE,max))',
    'MeanOccupancyPerActiveCU': 'reduce(accumulate(SQ_LEVEL_WAVES,LOW_RES),sum)/reduce(SQ_BUSY_CU_CYCLES,sum)',
})
DEPENDENCIES = {n: [n] for n in RAW}
DEPENDENCIES.update({
    'VALU_BUSY_PERCENT': ['SQ_ACTIVE_INST_VALU', 'GRBM_GUI_ACTIVE'],
    'MFMA_BUSY_PERCENT': ['SQ_VALU_MFMA_BUSY_CYCLES', 'GRBM_GUI_ACTIVE'],
    'MeanOccupancyPerActiveCU': ['MeanOccupancyPerActiveCU'],
})


def counter_name(name):
    return 'CSB_' + name if name in EXPRESSIONS else name


def canonical(name):
    return name[4:] if name.startswith('CSB_') and name[4:] in EXPRESSIONS else name


def names(group):
    return [counter_name(n) for n in GROUPS[group]]


def dependencies(group):
    return sorted({d for n in GROUPS[group] for d in DEPENDENCIES.get(n, [n])})


def extra_counters(path):
    # JSON is a YAML subset. No PyYAML dependency is needed on the target host.
    write(path, {'rocprofiler-sdk': {'counters-schema-version': 1, 'counters': [
        {'name': counter_name(n), 'description': f'MI300X benchmark: {n}; per-dispatch gfx942 reduction',
         'properties': [], 'definitions': [{'architectures': ['gfx942'], 'expression': expr}]}
        for n, expr in sorted(EXPRESSIONS.items())]}})
    return path


def definitions():
    return {'sources': SOURCES, 'expressions': EXPRESSIONS, 'groups': GROUPS,
            'ipc': 'sum(SQ_INSTS)/sum(SQ_BUSY_CU_CYCLES), ROCm Compute Profiler gfx942 convention',
            'occupancy_percent': '100*MeanOccupancyPerActiveCU/32; 4 SIMD/CU, 8 wave64/SIMD',
            'instruction_attribution': 'F32 FMA and ALL VALU type conversions, issued wave instructions. CVT is not an I2FP-only opcode counter.',
            'scratch': 'TA buffer read/write wavefront instructions: spill/stack activity, not spill bytes or a register-spill-only count.',
            'mfma': '512*MOPS; issued matrix work assumes full EXEC, separate I8 and F16 domains.',
            'register_limit': 'min(8,floor(512/(8*ceil((VGPR+AGPR)/8)))) waves/SIMD; vector-register-only upper bound, not achieved occupancy.'}
