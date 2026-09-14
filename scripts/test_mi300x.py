"""Offline MI300X protocol/isolation tests. No HIP, GPU, SDK or model download."""
import contextlib
import csv
import io
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mi300x as cli
from mi300x import common, instrumentation, profiling, report, serving, setup


class MI300XTests(unittest.TestCase):
    def cfg(self, **changes):
        return dict(backend='rocm', model_size='70b', quant='Q2_K', model_path='/models/q2.gguf',
                    model_sha256='abc', prompt_tokens=2048, output_tokens=512, concurrency=2, requests=4,
                    devices=['2', '5'], split=[1, 1], batch=2048, ubatch=512, kv_type='f16', seed=1,
                    chat_template_file=None, **changes)

    def test_amd_environment_is_child_local_and_removes_both_profiler_stacks(self):
        inherited = {'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '7', 'CUDA_HOME': '/cuda',
                     'HIP_VISIBLE_DEVICES': '9', 'ROCR_VISIBLE_DEVICES': '10', 'LD_PRELOAD': '/nvidia.so',
                     'LD_LIBRARY_PATH': '/cuda/lib', 'GPU_DEVICE_ORDINAL': '11', 'GGML_CUDA_FORCE_MMQ': '1',
                     'ROCP_TOOL_LIBRARIES': '/other.so', 'ROCM_TOOL_LIBRARIES': '/also.so',
                     'CXX': 'nvcc', 'LLAMA_ARG_MODEL': '/other.gguf'}
        with patch.dict(os.environ, inherited, clear=True):
            env = common.environment(['2', '5'], Path('/opt/rocm'))
            self.assertEqual(dict(os.environ), inherited)
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '2,5')
        self.assertEqual(env['HIP_PLATFORM'], 'amd')
        for name in inherited.keys() - {'PATH', 'HIP_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'}:
            self.assertNotIn(name, env)
        self.assertNotIn('/cuda', env['LD_LIBRARY_PATH'])

    def test_plan_does_not_probe_hardware_or_write_and_handles_64k(self):
        out = io.StringIO()
        with patch.object(serving, 'devices_info', side_effect=AssertionError('GPU call')), \
             patch.object(common, 'write', side_effect=AssertionError('write')), contextlib.redirect_stdout(out):
            code = cli.main(['plan', '--devices', '0', '1', '--formats', 'Q2_K', '--contexts', '65536',
                             '--concurrency', '8', '16', '32'])
        self.assertEqual(code, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['cell_count'], 3)
        self.assertAlmostEqual(sum(plan['jobs'][0]['kv_gib_per_gpu']), 161.875)
        self.assertEqual([j['concurrency'] for j in plan['jobs']], [8, 16, 32])

    def test_no_runtime_import_of_cuda_or_nvidia_tools(self):
        code = '''import sys
sys.path.insert(0, 'scripts')
import run_mi300x
run_mi300x.parser()
assert not any(k in sys.modules for k in ('profile_cuda', 'matrix_diagnostic', 'torch', 'pynvml', 'ncu_report'))
'''
        subprocess.run([sys.executable, '-c', code], cwd=common.ROOT, check=True)

    def test_baseline_flags_and_rocm_positive_placement(self):
        cfg = self.cfg()
        command = serving.server_command(cfg, 18080)
        self.assertEqual(command[0], str(common.STATE / 'build/bin/llama-server'))
        self.assertEqual(command[command.index('--ctx-size') + 1], '5632')
        self.assertIn('--no-context-shift', command)
        self.assertNotIn('rocprofv3', command)
        log = ('offloaded 81/81 layers to GPU\nn_seq_max = 2\nn_ctx_per_seq = 2816\n'
               'ROCm0 KV buffer size = 111.00 MiB\nROCm1 KV buffer size = 112.00 MiB\n'
               'K (f16): 100 MiB, V (f16): 100 MiB\n')
        geometry = {'layers': 80, 'layer_devices': [0] * 41 + [1] * 40}
        self.assertTrue(serving.placement(log, geometry, cfg)['valid'])
        for bad in [log.replace('ROCm', 'CUDA'), log.replace('81/81', '80/81'),
                    log.replace('K (f16)', 'K (q8_0)'), log + 'CPU KV buffer size = 1 MiB\n']:
            self.assertFalse(serving.placement(bad, geometry, cfg)['valid'])

    def test_output_marker_rejects_existing_cuda_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common.write(root / 'cell.json', {'backend': 'cuda'})
            with self.assertRaisesRegex(RuntimeError, 'unmarked'):
                common.output_root(root)
            self.assertEqual(common.read(root / 'cell.json'), {'backend': 'cuda'})

    def test_target_shutdown_allows_profiler_parent_to_flush(self):
        # Real CPU subprocesses: an HTTP target plus a parent acting as profiler.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                port = s.getsockname()[1]
            target = root / 'target.py'
            target.write_text('''import http.server, os, pathlib, signal, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
server = http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
pathlib.Path(os.environ['CSB_MI300X_GATE'] + '.ready').write_text(str(os.getpid()))
signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
server.serve_forever()
''')
            wrapper = root / 'wrapper.py'
            wrapper.write_text('''import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
code = child.wait()
pathlib.Path(sys.argv[3]).write_text('flushed')
sys.exit(code)
''')
            env = dict(os.environ, CSB_MI300X_GATE=str(root / 'gate'))
            with common.server([sys.executable, str(wrapper), str(target), str(port), str(root / 'flushed')],
                               env, root, port, 5) as process:
                self.assertIsNone(process.poll())
            self.assertEqual(process.returncode, 0)
            self.assertEqual((root / 'flushed').read_text(), 'flushed')

    def test_probe_rejects_warmup_leakage_and_missing_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {'Kernel_Name': 'csb_mi300x_probe(float*)', 'Counter_Name': 'SQ_WAVES', 'Counter_Value': '64'}
            self.fixture_csv(root / 'probe_counter_collection.csv', [row])
            self.fixture_csv(root / 'probe_kernel_trace.csv', [{'Kernel_Name': 'csb_mi300x_probe(float*)'}])
            self.assertTrue(profiling.validate_probe(root))
            self.fixture_csv(root / 'probe_counter_collection.csv', [dict(row, Counter_Value='4'), row])
            self.assertFalse(profiling.validate_probe(root))
            self.fixture_csv(root / 'probe_counter_collection.csv', [dict(row, Counter_Value='4')])
            self.assertFalse(profiling.validate_probe(root))

    def test_serving_completion_resume_and_changed_hardware_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = cli.parser().parse_args(['serve', '--formats', 'Q2_K', '--contexts', '2048',
                                           '--concurrency', '2', '--repetitions', '1', '--output', str(root)])
            args.devices, args.split, args.port = ['0', '1'], [1, 1], 18080
            row = dict(quant='Q2_K', path='/fixture.gguf', sha256='abc', bytes=100, model_size='70b')
            gpu = {'name': 'AMD Instinct MI300X', 'total_bytes': 192 * 1024**3, 'free_bytes': 190 * 1024**3}
            info = {'devices': [dict(gpu, physical_hip_index='0', pci_bus='0000:01:00.0'),
                                dict(gpu, physical_hip_index='1', pci_bus='0000:02:00.0')]}
            geometry = dict(available=True, layers=80, layer_devices=[0] * 41 + [1] * 40,
                            kv_heads=8, head_dimension=128, weight_bytes_per_gpu=[50, 50])

            @contextlib.contextmanager
            def fake_server(command, env, directory, port, timeout):
                (directory / 'server.log').write_text('fixture')
                yield SimpleNamespace(pid=os.getpid())

            @contextlib.contextmanager
            def monitor(pid):
                yield SimpleNamespace(peak=0, error=None)

            with patch.object(serving, 'prepared', return_value=[row]), \
                 patch.object(serving, 'check_build', return_value={'backend': 'rocm', 'artifacts': {}}), \
                 patch.object(serving, 'devices_info', return_value=info), \
                 patch.object(serving, 'inspect_model', return_value=geometry), \
                 patch.object(serving, 'server', side_effect=fake_server) as launch, \
                 patch.object(serving, 'placement', return_value={'valid': True}), \
                 patch.object(serving, 'make_prompts', return_value=[]), \
                 patch.object(serving, 'SwapMonitor', side_effect=monitor), \
                 patch.object(serving, 'burst', return_value={'valid': True, 'summary': {'output_tokens_per_second': 1}}):
                self.assertEqual(serving.serve(args), [])
                self.assertEqual(launch.call_count, 1)
                cell = root / 'cells/70b/Q2_K/p2048/c2/r1/cell.json'
                self.assertEqual(common.read(cell)['status'], 'completed')
                original = cell.read_bytes()
                args.resume = True
                self.assertEqual(serving.serve(args), [])
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(original, cell.read_bytes())
                args.devices = ['2', '3']
                with self.assertRaisesRegex(RuntimeError, 'different serving'):
                    serving.serve(args)

    def test_private_annotations_apply_to_exact_pin_without_modifying_source(self):
        vendor = common.ROOT / 'vendor/llama.cpp'
        if not (vendor / '.git').exists():
            self.skipTest('Pinned source checkout not installed')
        relative = 'ggml/src/ggml-cuda/ggml-cuda.cu'
        old = subprocess.check_output(['git', '-C', str(vendor), 'show', f'{common.COMMIT}:{relative}'], text=True)
        updated = instrumentation.annotate_operator(old)
        self.assertEqual(updated.count('csb_fused_matrix_range csb_fused_range('), 9)
        self.assertIn('roctxRangePushA', updated)
        self.assertNotIn('nvtxRangePushA', updated)
        self.assertNotIn('CSB_NVTX_OPS', updated)
        old_server = subprocess.check_output(['git', '-C', str(vendor), 'show',
                                             f'{common.COMMIT}:tools/server/server-context.cpp'], text=True)
        updated_server = instrumentation.annotate_server(old_server)
        self.assertIn('batch.tokens[i].is_prompt', updated_server)
        self.assertIn('"mixed"', updated_server)
        with self.assertRaisesRegex(RuntimeError, 'insertion'):
            instrumentation.annotate_operator(updated)

    def test_burst_validates_complete_concurrency_and_failure(self):
        cfg = self.cfg()
        prompts = [dict(messages=[], prompt_tokens=2048, input_sha256=str(i)) for i in range(4)]

        def stream(*args, **kwargs):
            start = time.perf_counter()
            time.sleep(.025)
            return {'ok': True, '_started': start, '_finished': time.perf_counter(),
                    'started_unix_seconds': time.time(), 'completion_tokens': 512,
                    'ttft_seconds': .01, 'tpot_seconds': .001, 'duration_seconds': .025}

        with patch.object(serving.load_test, 'stream_completion', side_effect=stream), \
             patch.object(serving.load_test, 'finalize_record') as finalize:
            result = serving.burst(cfg, prompts, 'http://fake', 2)
        self.assertTrue(result['valid'])
        self.assertEqual(result['summary']['max_client_inflight'], 2)
        self.assertEqual(result['summary']['successful_requests'], 4)
        self.assertEqual(finalize.call_args.kwargs['expected_prompt_tokens'], 2048)
        self.assertEqual(finalize.call_args.kwargs['cache_policy'], 'forbid')

    def fixture_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)

    def capture_fixture(self, directory, duplicate=False):
        root = directory / 'rocprof'
        (directory / 'gate.ready').write_text('123')
        self.fixture_csv(root / 'capture_hip_api_trace.csv', [
            {'Function': 'hipLaunchKernel', 'Process_Id': '123', 'Thread_Id': '7',
             'Correlation_Id': '9', 'Start_Timestamp': '30', 'End_Timestamp': '31'},
            {'Function': 'hipLaunchKernel', 'Process_Id': '456', 'Thread_Id': '7',
             'Correlation_Id': '9', 'Start_Timestamp': '30', 'End_Timestamp': '31'}])
        self.fixture_csv(root / 'capture_marker_api_trace.csv', [
            {'Function': 'csb_batch:phase=decode:width=8', 'Process_Id': '123', 'Thread_Id': '7',
             'Start_Timestamp': '10', 'End_Timestamp': '50'},
            {'Function': 'csb_op:role=blk.0.ffn_up.weight:type=q2_K:m=28672:n=8:k=8192',
             'Process_Id': '123', 'Thread_Id': '7', 'Start_Timestamp': '20', 'End_Timestamp': '40'}])
        # GPU execution happens AFTER the CPU ranges have ended. Counter CSVs in
        # current SDKs can omit Process_Id and use three-dimensional grid fields.
        base = {'Kernel_Name': 'mul_mat_q', 'Dispatch_Id': '1', 'Agent_Id': '4', 'Thread_Id': '7',
                'Correlation_Id': '9', 'Start_Timestamp': '100', 'End_Timestamp': '1100',
                'Grid_Size_X': '16', 'Grid_Size_Y': '1', 'Grid_Size_Z': '1',
                'Workgroup_Size_X': '256', 'Workgroup_Size_Y': '1', 'Workgroup_Size_Z': '1'}
        rows = [dict(base, Counter_Name=n, Counter_Value=str(v)) for n, v in {
            'SQ_INSTS_VALU_ADD_F32': 1, 'SQ_INSTS_VALU_MUL_F32': 2, 'SQ_INSTS_VALU_FMA_F32': 3,
            'SQ_INSTS_VALU_MFMA_MOPS_F16': 2, 'FETCH_SIZE': 3, 'WRITE_SIZE': 1}.items()]
        if duplicate:
            rows.append(dict(rows[-1], Counter_Dimension='1'))
        self.fixture_csv(root / 'capture_counter_collection.csv', rows)

    def test_async_operator_attribution_and_same_dispatch_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture_fixture(directory)
            data = report.normalize_capture(directory)
        self.assertEqual(len(data['kernels']), 1)
        k = data['kernels'][0]
        self.assertEqual(k['phase'], 'decode')
        self.assertEqual(k['role'], 'blk.0.ffn_up.weight')
        self.assertEqual(k['agent'], '4')  # Raw agent ID, never rewritten as HIP index.
        self.assertEqual(k['grid'], '16x1x1')
        fp32 = report.coordinates(k, 'scalar_fp32')
        self.assertEqual(fp32['operations'], 576)
        self.assertEqual(fp32['dram_bytes'], 4096)
        self.assertAlmostEqual(fp32['duration_s'], 1e-6)
        self.assertEqual(report.coordinates(k, 'mfma_f16')['operations'], 1024)

    def test_duplicate_counter_instances_are_unavailable_not_summed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture_fixture(directory, duplicate=True)
            k = report.normalize_capture(directory)['kernels'][0]
        self.assertIsNone(k['metrics']['WRITE_SIZE'])
        with self.assertRaisesRegex(ValueError, 'Missing/ambiguous'):
            report.coordinates(k, 'mfma_f16')

    def test_counter_operands_never_join_different_pass_files_or_agents(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.capture_fixture(directory)
            first = directory / 'rocprof/capture_counter_collection.csv'
            rows = list(profiling.csv_rows(first))
            self.fixture_csv(first, rows[:-1])
            self.fixture_csv(directory / 'rocprof/other_counter_collection.csv', rows[-1:])
            self.fixture_csv(directory / 'rocprof/third_counter_collection.csv',
                             [dict(r, Agent_Id='8') for r in rows[:-1]])
            kernels = report.normalize_capture(directory)['kernels']
        self.assertEqual(len(kernels), 3)
        for k in kernels:
            with self.assertRaises(ValueError):
                report.coordinates(k, 'scalar_fp32')

    def test_report_exports_missing_cases_and_does_not_plot_fake_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = common.output_root(Path(temporary))
            common.write(root / 'cells/one/cell.json', {'backend': 'rocm', 'configuration': self.cfg(),
                'status': 'capacity_estimated', 'repetition': 1, 'summary': {'ttft_p95_ms': 1}})
            args = SimpleNamespace(output=root, no_plots=True, peak_fp32_tflops=163.4,
                                   peak_mfma_tflops=1307.4, peak_int8_tops=2614.9, bandwidth_tb_s=5.3)
            report.report(args)
            rows = list(profiling.csv_rows(root / 'reports/serving.csv'))
            self.assertEqual(rows[0]['status'], 'capacity_estimated')
            self.assertNotIn('ttft_p95_ms', rows[0])
            self.assertFalse((root / 'reports/runtime-cost-merged.pdf').exists())

    def test_report_renders_latency_throughput_and_separate_arithmetic_roofs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = common.output_root(Path(temporary))
            cfg = self.cfg()
            common.write(root / 'cells/one/cell.json', {'backend': 'rocm', 'configuration': cfg,
                'status': 'completed', 'repetition': 1,
                'summary': {'ttft_p95_ms': 100, 'tpot_p95_ms': 20, 'output_tokens_per_second': 50}})
            directory = root / 'profiles/one/scalar_fp32'
            directory.mkdir(parents=True)
            self.capture_fixture(directory)
            common.write(directory / 'capture.json', {'backend': 'rocm', 'configuration': cfg,
                                                     'status': 'captured', 'kind': 'scalar_fp32'})
            common.write(directory / 'kernels.json', report.normalize_capture(directory))
            args = SimpleNamespace(output=root, no_plots=False, peak_fp32_tflops=163.4,
                                   peak_mfma_tflops=1307.4, peak_int8_tops=2614.9, bandwidth_tb_s=5.3)
            report.report(args)
            for name in ('runtime-cost-merged.pdf', 'throughput-context.pdf', 'roofline-explained.pdf'):
                self.assertTrue((root / 'reports' / name).read_bytes().startswith(b'%PDF'))
            rows = list(profiling.csv_rows(root / 'reports/roofline-values.csv'))
            self.assertEqual(len(rows), 1)
            self.assertIn('proxy', rows[0]['definition'])

    def test_prepare_does_not_change_shared_manifest_or_download_local_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.gguf'
            model.write_bytes(b'GGUFfixture')
            manifest_path = root / 'models.json'
            common.write(manifest_path, [dict(model_size='70b', quant='Q2_K', filename='model.gguf',
                                             path=str(model), bytes=model.stat().st_size, sha256=common.sha(model))])
            original = manifest_path.read_bytes()
            args = SimpleNamespace(manifest=manifest_path, model_size='70b', formats=['Q2_K'], model_dir=root / 'amd')
            with patch.object(setup, 'STATE', root / 'state'):
                setup.prepare(args)
            self.assertEqual(manifest_path.read_bytes(), original)
            self.assertEqual(model.read_bytes(), b'GGUFfixture')
            model.unlink()
            with patch.object(setup, 'STATE', root / 'state'), self.assertRaisesRegex(RuntimeError, 'supplied locally'):
                setup.prepare(args)


if __name__ == '__main__':
    unittest.main()
