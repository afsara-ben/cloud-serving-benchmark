"""Offline acceptance tests for AMD collection scope and publication evidence."""
import copy
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mi300x as cli
from mi300x import analysis, common, metrics, profiling, publication, report


def arguments(root, devices=('0', '1')):
    args = cli.parser().parse_args(['paper', '--output', str(root), '--devices', *devices,
                                   '--contexts', '2048', '--profile-tag', 'fixture'])
    args.split = [1] * len(devices)
    return args


def fixture_cell(root, q='Q2_K', c=8, p=2048, devices=('0', '1')):
    args = arguments(root, devices)
    cfg = dict(backend='rocm', model_size='70b', quant=q, prompt_tokens=p, output_tokens=512,
               concurrency=c, requests=2*c, devices=args.devices, split=args.split, kv_type='f16',
               seed=1, batch=2048, ubatch=512, model_path='/fixture.gguf', model_sha256='fixture')
    path = root / f'cells/70b/{q}/p{p}/c{c}/r1/cell.json'
    prompts = []
    for i in range(2*c):
        messages = [{'role': 'user', 'content': f'test fixture {i}'}]
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        prompts.append(dict(messages=messages, prompt_tokens=p, input_sha256=digest))

    def burst(count, outputs):
        records = []
        for i in range(count):
            start, duration, ttft = (i // c) * 10, 10., 2.
            records.append(dict(ok=True, error=None, completion_tokens=outputs, prompt_tokens=p,
                                evaluated_prompt_tokens=p, cached_prompt_tokens=0, input_sha256=prompts[i]['input_sha256'],
                                start_offset_seconds=start, end_offset_seconds=start+duration,
                                ttft_seconds=ttft, tpot_seconds=(duration-ttft)/(outputs-1), duration_seconds=duration,
                                stream_done_received=True, stream_content_events=outputs, finish_reason='length',
                                server_usage={'prompt_tokens': p, 'completion_tokens': outputs, 'prompt_tokens_details': {'cached_tokens': 0}},
                                server_timings={'prompt_n': p, 'predicted_n': outputs, 'cache_n': 0}))
        return {'valid': True, 'requests': records}
    common.write(path.parent / 'prompts.json', prompts)
    common.write(path.parent / 'requests.json', burst(2*c, 512))
    common.write(path.parent / 'warmup.json', burst(c, 16))
    common.write(path, {'backend': 'rocm', 'status': 'completed', 'configuration': cfg, 'repetition': 1,
                       'placement': {'valid': True}, 'peak_process_swap_bytes': 0,
                       'gpu_identity': [{'physical_hip_index': i, 'pci_bus': i, 'name': 'MI300X'} for i in devices],
                       'summary': {'output_tokens_per_second': 2*c*512/20, 'ttft_p95_ms': 2000, 'tpot_p95_ms': 8000/511}})
    return path, cfg


def fixture_kernel(q='Q2_K', phase='prefill', gpu=0, index=0, family='gate', fused=False):
    typ = 'q2_K' if q == 'Q2_K' else 'q4_K'
    role = f'blk.{index}.ffn_{family}.weight' if family in ('gate', 'up') else f'blk.{index}.attn_q.weight'
    op = dict(type=typ, m='28672', n='512' if phase == 'prefill' else '8', k='8192')
    if fused:
        role = f'blk.{index}.ffn_up.weight+blk.{index}.ffn_gate.weight'
        op.update(type=typ+'+'+typ, fusion='gate_up_glu', matrix_count='2')
    if family == 'Q':
        op['m'] = '8192'
    m = {'FETCH_SIZE': 80 if q == 'Q2_K' else 100, 'WRITE_SIZE': 20,
         'SQ_INSTS': 2000 if q == 'Q2_K' else 1000,
         'SQ_INSTS_VALU_FMA_F32': 800 if q == 'Q2_K' else 200,
         'SQ_INSTS_VALU_CVT': 400 if q == 'Q2_K' else 100,
         'SQ_BUSY_CU_CYCLES': 500, 'VALU_BUSY_PERCENT': 40, 'MFMA_BUSY_PERCENT': 20,
         'MeanOccupancyPerActiveCU': 16, 'TA_BUFFER_READ_WAVEFRONTS_sum': 0, 'TA_BUFFER_WRITE_WAVEFRONTS_sum': 0,
         'SQ_INSTS_VALU_MFMA_MOPS_I8': 10 if phase == 'prefill' else 0,
         'SQ_INSTS_VALU_MFMA_MOPS_F16': 0, 'SQ_INSTS_VALU_ADD_F32': 10, 'SQ_INSTS_VALU_MUL_F32': 20}
    return dict(file='counter.csv', process='123', agent=str(400+gpu), hip_device=str(gpu), dispatch=str(index),
                kernel='mul_mat_'+q, grid='32x1x1', workgroup='256x1x1', role=role, operation=op,
                phase=phase, start_ns=index*100, duration_ns=1000 if q == 'Q4_K_M' else 1500,
                vgpr_count='120', agpr_count='8', sgpr_count='32', lds_bytes='1024', scratch_bytes='0',
                metrics=m, ambiguous_metrics=[])


def fixture_study(root, devices=('0', '1')):
    common.output_root(root)
    args = arguments(root, devices)
    cells = {}
    for q in args.formats:
        for c in args.concurrency:
            cells[(q, c)] = fixture_cell(root, q, c, devices=devices)
    for q in ('Q2_K', 'Q4_K_M'):
        source, cfg = cells[(q, 8)]
        for phase in ('prefill', 'decode'):
            for group in ('trace', *metrics.BOTTLE_GROUPS):
                directory = root / f'profiles/fixture/70b/{q}/p2048/c8/r1/{phase}/{group}'
                kernels = []
                for gpu in range(len(devices)):
                    kernels.extend(fixture_kernel(q, phase, gpu, i, family='gate' if i % 2 == 0 else 'up') for i in range(10))
                    kernels.extend(fixture_kernel(q, phase, gpu, 20+i, family='Q') for i in range(5))
                common.write(directory / 'kernels.json', {'backend': 'rocm', 'kernels': kernels,
                                                         'coverage': {'dispatches': len(kernels), 'with_operator_and_phase': len(kernels)}})
                raw = directory / 'rocprof/test-fixture.csv'; raw.parent.mkdir()
                raw.write_text('synthetic unit-test fixture; no GPU measurement\n')
                common.write(directory / 'capture.json', {'backend': 'rocm', 'status': 'captured',
                    'kind': group, 'configuration': cfg, 'capture_phase': phase, 'capture_batches': 1,
                    'captured_batches': 1, 'build': {'backend': 'rocm', 'artifact': 'fixture'},
                    'protocol': {'source': 'fixture'}, 'source_cell': str(source), 'source_cell_sha256': common.sha(source),
                    'prompts_sha256': common.sha(source.parent / 'prompts.json'),
                    'kernels_sha256': common.sha(directory / 'kernels.json'),
                    'raw_files': {'rocprof/test-fixture.csv': common.sha(raw)}})
    return args


def fixture_update_kernels(path, data):
    common.write(path, data)
    meta_path = path.parent / 'capture.json'
    meta = common.read(meta_path)
    meta['kernels_sha256'] = common.sha(path)
    common.write(meta_path, meta)


class PublicationTests(unittest.TestCase):
    def test_int8_domain_and_instruction_reconciliation(self):
        k = fixture_kernel()
        self.assertEqual(report.coordinates(k, 'mfma_i8')['operations'], 5120)
        self.assertEqual(report.coordinates(k, 'mfma_f16')['operations'], 0)
        v = analysis.values_for(k, 'attribution')
        self.assertEqual(v['instructions'], v['fma_f32'] + v['conversions'] + v['other_instructions'])
        self.assertEqual(analysis.values_for(k, 'ipc')['ipc'], 4)
        self.assertEqual(analysis.values_for(k, 'occupancy')['occupancy_percent'], 50)
        k['metrics']['SQ_INSTS'] = 10
        with self.assertRaisesRegex(ValueError, 'exceed'):
            analysis.values_for(k, 'attribution')

    def test_register_bound_includes_accumulator_registers(self):
        k = fixture_kernel()
        k.update(vgpr_count='125', agpr_count='48')
        v = analysis.resources(k)
        self.assertEqual(v['vector_registers_allocated'], 176)
        self.assertEqual(v['register_wave_limit_per_simd'], 2)
        self.assertEqual(v['register_workgroup_bound_per_cu'], 2)
        k['agpr_count'] = None
        self.assertNotIn('register_wave_limit_per_simd', analysis.resources(k))

    def test_custom_counter_schema_reduces_arrays_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = metrics.extra_counters(Path(temporary) / 'extra.yaml')
            counters = common.read(path)['rocprofiler-sdk']['counters']
            lookup = {c['name']: c['definitions'][0] for c in counters}
            self.assertEqual(lookup['CSB_SQ_INSTS']['expression'], 'reduce(SQ_INSTS,sum)')
            self.assertEqual(lookup['CSB_SQ_INSTS']['architectures'], ['gfx942'])
            self.assertIn('SQ_VALU_MFMA_BUSY_CYCLES', metrics.dependencies('activity'))
            self.assertEqual(metrics.canonical('CSB_SQ_INSTS'), 'SQ_INSTS')

    def test_profiler_gpu_mapping_uses_pci_not_hip_or_raw_agent_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            common.write(directory / 'probe.json', {'tool': [{'agents': [
                {'id': {'handle': 800}, 'logical_node_type_id': 7, 'domain': 0, 'location_id': 0x4100},
                {'id': {'handle': 900}, 'logical_node_type_id': 9, 'domain': 0, 'location_id': 0x4200}]}]})
            self.assertEqual(profiling.profiler_device_index(directory, '0000:42:00.0'), '9')
            with self.assertRaisesRegex(RuntimeError, 'unique profiler'):
                profiling.profiler_device_index(directory, '0000:43:00.0')

    def test_doctor_uses_global_device_option_and_probes_custom_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            args = cli.parser().parse_args(['doctor', '--devices', '2', '--groups', 'attribution'])
            commands = []
            def fake_run(command, **kwargs):
                command = list(map(str, command)); commands.append(command)
                if command[0] == 'avail':
                    self.assertEqual(command[1:3], ['-d', '7'])
                return SimpleNamespace(returncode=0)
            def rows(path):
                if path.name.endswith('counter_collection.csv'):
                    return iter([{'Counter_Name': metrics.counter_name(n), 'Counter_Value': '1'} for n in metrics.GROUPS['attribution']])
                return iter([])
            def probe_run(command, **kwargs):
                fake_run(command, **kwargs)
                if '--output-directory' in command:
                    p = Path(command[command.index('--output-directory')+1]);p.mkdir(parents=True, exist_ok=True)
                    (p / 'probe_counter_collection.csv').touch()
            with patch.object(profiling, 'STATE', state), patch.object(profiling, 'check_build'), \
                 patch.object(profiling, 'tool', side_effect=lambda n,e: 'avail' if n.endswith('-avail') else 'profiler'), \
                 patch.object(profiling, 'devices_info', return_value={'devices': [{'physical_hip_index':'2','pci_bus':'0000:42:00.0'}]}), \
                 patch.object(profiling.subprocess, 'check_output', return_value='--selected-regions --preload --kernel-iteration-range --kernel-include-regex --hip-trace --kernel-trace --marker-trace --memory-copy-trace --pmc --extra-counters'), \
                 patch.object(profiling.subprocess, 'run', side_effect=fake_run), \
                 patch.object(profiling, 'run', side_effect=probe_run), \
                 patch.object(profiling, 'validate_probe', return_value=True), \
                 patch.object(profiling, 'profiler_device_index', return_value='7'), \
                 patch.object(profiling, 'csv_rows', side_effect=rows):
                result = profiling.doctor(args)
            self.assertTrue(result['counter_groups']['2']['attribution']['available'])
            self.assertTrue(any('CSB_SQ_INSTS_VALU_CVT' in c and '--extra-counters' in c for c in commands))

    def test_selection_excludes_helpers_wrong_phase_shape_and_type(self):
        rows = [fixture_kernel(index=i, family='gate' if i % 2 == 0 else 'up') for i in range(10)]
        decoy = fixture_kernel(index=-1); decoy['kernel'] = 'activation_quantize'
        wrong = fixture_kernel(index=-2); wrong['operation']['n'] = '8'
        wrong_type = fixture_kernel(index=-3); wrong_type['operation']['type'] = 'q6_K'
        candidates = analysis.gate_candidates([decoy, wrong, wrong_type, *rows], 'Q2_K', 'prefill', 512)
        selected, ref = analysis.select_samples(candidates, 5)
        self.assertEqual([r['dispatch'] for r in selected], ['0', '2', '4', '1', '3'])
        with self.assertRaises(ValueError):
            analysis.select_samples([dict(r, workgroup='128') for r in candidates], 5, ref)

    def test_fused_samples_are_whole_kernels(self):
        rows = [fixture_kernel(index=i, fused=True) for i in range(8)]
        selected, _ = analysis.select_samples(rows, 5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(k['operation']['matrix_count'] == '2' for k in selected))

    def test_poster_revalidates_requests_and_breaks_missing_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, cfg = fixture_cell(root)
            value = analysis.validate_serving(path)
            self.assertEqual(value['output_tokens_per_second'], 409.6)
            records = common.read(path.parent / 'requests.json')
            records['requests'][0]['cached_prompt_tokens'] = 1
            common.write(path.parent / 'requests.json', records)
            with self.assertRaises(ValueError):
                analysis.validate_serving(path)
            args = arguments(root)
            data = publication.serving_data(root, args)
            row = next(r for r in data['coverage'] if r['quant'] == 'Q2_K' and r['concurrency'] == 8)
            self.assertEqual(row['status'], 'incomplete')
            self.assertNotIn('output_tokens_per_second', row)

    def test_full_evidence_grid_and_exact_extra_instruction_share(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            result = analysis.analyze_captures(root, args)
            self.assertTrue(result['complete'], result['errors'])
            self.assertEqual(len(result['medians']), 80)
            self.assertEqual(len(result['selections']), 400)
            comparisons, shares = publication.comparisons(result, args)
            self.assertEqual(len(shares), 4)
            self.assertTrue(all(s['share_percent'] == 90 for s in shares))
            self.assertTrue(publication.serving_data(root, args)['complete'])

    def test_missing_gpu_and_changed_fusion_are_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            path = root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/decode/memory/kernels.json'
            data = common.read(path)
            data['kernels'] = [k for k in data['kernels'] if k['hip_device'] == '0']
            fixture_update_kernels(path, data)
            result = analysis.analyze_captures(root, args)
            self.assertFalse(result['complete'])
            self.assertTrue(any('Q2_K/decode/GPU1' in e for e in result['errors']))

    def test_rejects_mixed_builds_and_changed_serving_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            path = root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/decode/memory/capture.json'
            data = common.read(path); data['build']['artifact'] = 'other'
            common.write(path, data)
            with self.assertRaisesRegex(ValueError, 'different diagnostic'):
                analysis.analyze_captures(root, args)

    def test_changed_raw_counter_file_invalidates_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            raw = root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/prefill/memory/rocprof/test-fixture.csv'
            raw.write_text('changed after capture\n')
            result = analysis.analyze_captures(root, args)
            self.assertFalse(result['complete'])
            self.assertTrue(any('raw or normalized counter evidence missing/changed' in e for e in result['errors']))

    def test_missing_counter_group_fails_coverage_but_preserves_other_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            path = root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/decode/attribution/capture.json'
            data = common.read(path); data['status'] = 'unsupported_counter_group'
            common.write(path, data)
            result = analysis.analyze_captures(root, args)
            self.assertFalse(result['complete'])
            self.assertTrue(any(r['group'] == 'memory' for r in result['medians']))

    def test_zero_hbm_helpers_are_recorded_without_infinite_roofline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            path = root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/decode/scalar_fp32/kernels.json'
            data = common.read(path)
            for k in data['kernels']:
                if analysis.family(k) == 'Q':
                    k['metrics'].update(FETCH_SIZE=0, WRITE_SIZE=0)
            fixture_update_kernels(path, data)
            result = analysis.analyze_captures(root, args)
            self.assertTrue(result['complete'], result['errors'])
            self.assertTrue(any(r['status'] == 'zero_hbm_traffic' for r in result['operator_coverage']))

    def test_mismatched_fusion_cannot_produce_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            for path in (root / 'profiles/fixture/70b/Q2_K/p2048/c8/r1/decode').rglob('kernels.json'):
                data = common.read(path)
                for k in data['kernels']:
                    if analysis.family(k) in ('gate', 'up'):
                        k['role'] += '+blk.0.ffn_up.weight+blk.0.ffn_gate.weight'
                        k['operation'].update(fusion='gate_up_glu', matrix_count='2', type='q2_K+q2_K')
                fixture_update_kernels(path, data)
            result = analysis.analyze_captures(root, args)
            self.assertFalse(result['complete'])
            self.assertTrue(any('different fusion scopes' in e for e in result['errors']))

    def test_counter_geometry_joins_only_the_same_dispatch_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); raw = directory / 'rocprof'; raw.mkdir()
            base = dict(Kernel_Name='mul_mat_q', Process_Id='99', Agent_Id='12', Dispatch_Id='7',
                        Start_Timestamp='10', End_Timestamp='100')
            counter = dict(base, Grid_Size='768', Workgroup_Size='128', Counter_Name='CSB_SQ_INSTS', Counter_Value='100')
            trace = dict(base, Grid_Size_X='256', Grid_Size_Y='3', Grid_Size_Z='1', Workgroup_Size_X='64',
                         Workgroup_Size_Y='2', Workgroup_Size_Z='1', Accum_VGPR_Count='32')
            for name, rows in [('a_counter_collection.csv', [counter]),
                               ('a_kernel_trace.csv', [trace, dict(trace, Agent_Id='13', Accum_VGPR_Count='64')])]:
                with (raw / name).open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            k = report.normalize_capture(directory)['kernels'][0]
            self.assertEqual((k['grid'], k['workgroup'], k['agpr_count']), ('256x3x1', '64x2x1', '32'))
            self.assertEqual(k['metrics']['SQ_INSTS'], 100)

    def test_publication_renders_both_documents_from_validated_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root)
            self.assertTrue(publication.build(args))
            directory = root / 'publication/fixture'
            for name in ('runtime-cost-poster.pdf', 'runtime-cost-merged.pdf', 'runtime-bottlenecks-report.pdf'):
                self.assertTrue((directory / name).read_bytes().startswith(b'%PDF'))
            self.assertTrue(common.read(directory / 'validation.json')['complete'])
            self.assertTrue((directory / 'operator-coverage.csv').is_file())

    def test_single_gpu_publication_needs_only_selected_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = fixture_study(root, devices=('0',))
            self.assertTrue(publication.build(args))
            directory = root / 'publication/fixture'
            kernels = common.read(directory / 'report-data.json')
            self.assertEqual(kernels['physical_gpus'], ['0'])
            self.assertEqual(len(kernels['medians']), 40)
            self.assertEqual(len(kernels['selections']), 200)
            self.assertEqual({r['gpu'] for r in kernels['medians']}, {'0'})
            self.assertEqual(len(common.read(directory / 'poster-data.json')['gpu_identity']), 1)
            self.assertTrue(common.read(directory / 'validation.json')['complete'])
            for name in ('runtime-cost-poster.pdf', 'runtime-cost-merged.pdf', 'runtime-bottlenecks-report.pdf'):
                self.assertTrue((directory / name).read_bytes().startswith(b'%PDF'))

    def test_incomplete_cli_returns_nonzero_with_reviewable_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = common.output_root(Path(temporary))
            result = cli.main(['paper', '--output', str(root), '--no-plots'])
            self.assertEqual(result, 2)
            validation = common.read(root / 'publication/run1/validation.json')
            self.assertFalse(validation['complete'])
            self.assertEqual(cli.main(['paper', '--output', str(root), '--no-plots', '--allow-incomplete']), 0)

    def test_phase_gate_arms_matches_width_limits_batches_and_restores_device(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable for CPU gate integration test')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'hip').mkdir(); (root / 'rocprofiler-sdk-roctx').mkdir()
            (root / 'hip/hip_runtime_api.h').write_text('''#pragma once
typedef int hipError_t;
static constexpr int hipSuccess=0;
static thread_local int selected_device=0;
inline int hipGetDevice(int*p){*p=selected_device;return 0;}
inline int hipGetDeviceCount(int*p){*p=2;return 0;}
inline int hipSetDevice(int d){selected_device=d;return 0;}
inline int hipDeviceSynchronize(){return 0;}
inline const char* hipGetErrorString(int){return "error";}
''')
            (root / 'rocprofiler-sdk-roctx/roctx.h').write_text('''#pragma once
#include <atomic>
static std::atomic<int> resumes{0}, pauses{0};
inline int roctxProfilerResume(int){++resumes;return 0;}
inline int roctxProfilerPause(int){++pauses;return 0;}
''')
            harness = root / 'gate-test.cpp'
            harness.write_text('#include "' + str(common.ROOT / 'benchmark/mi300x/gate.cpp') + '''"
extern "C" int counts(){return resumes*100+pauses;}
extern "C" int current_device(){return selected_device;}
''')
            subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-pthread', '-I', str(root), str(harness), '-o', str(root / 'gate.so')], check=True)
            # A child process keeps the test preload's detached watcher and
            # constructor state isolated from the unit-test runner.
            script = '''import ctypes,os,pathlib,time,sys
root=pathlib.Path(sys.argv[1]);prefix=root/'gate'
os.environ.update(CSB_MI300X_GATE=str(prefix),CSB_MI300X_PHASE='decode',CSB_MI300X_BATCHES='1',CSB_MI300X_WIDTH='8')
lib=ctypes.CDLL(str(root/'gate.so'))
def wait(suffix):
 deadline=time.time()+4
 while not prefix.with_suffix(suffix).exists():
  assert time.time()<deadline,suffix
  time.sleep(.01)
wait('.ready');prefix.with_suffix('.start').touch();wait('.started')
assert lib.counts()==0
assert lib.csb_mi300x_batch_begin(b'prefill',8)==0
assert lib.csb_mi300x_batch_begin(b'decode',4)==0
assert lib.csb_mi300x_batch_begin(b'decode',8)==1
assert lib.current_device()==0
lib.csb_mi300x_batch_end()
assert lib.counts()==101
assert lib.csb_mi300x_batch_begin(b'decode',8)==0
prefix.with_suffix('.stop').touch();wait('.stopped')
assert prefix.with_suffix('.batches').read_text()=='1'
assert lib.counts()==101
'''
            subprocess.run([sys.executable, '-c', script, str(root)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
