"""Offline checks for A100 section and roofline evidence validation."""
import copy
import unittest

import profile_cuda as profile


def sample(sections):
    names = {name for section in sections for name in profile.SECTION_REQUIREMENTS[section]}
    names.add("gpu__time_duration.sum")
    launch = {"id": "0", "device": "0", "metrics": {name: {"value": 1., "available": True} for name in names},
              "metric_instances": {"sass__inst_executed_per_opcode": [{"instance": "IADD3", "value": 10., "available": True}]}}
    return {"launches": [launch]}


class A100SectionTests(unittest.TestCase):
    def validate(self, summary, sections):
        return profile.validate_counter_capture(summary, ["gpu__time_duration.sum"], sections,
                                                 required_launches=1, expected_devices=["0"])

    def test_named_section_collection_retains_load_store_and_cache_transactions(self):
        collection = profile.a100_section_collection(["half", "tensor", "half"])
        self.assertEqual(len(collection["sections"]), 10)
        self.assertTrue(set(profile.A100_SECTIONS).issubset(collection["sections"]))
        self.assertIn("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", collection["metrics"].split(","))
        self.assertIn("lts__t_sectors_op_write.sum", collection["metrics"].split(","))
        with self.assertRaises(ValueError):
            profile.a100_section_collection(["int4"])

    def test_missing_compute_and_memory_section_evidence_is_rejected(self):
        sections = list(profile.A100_SECTIONS)
        valid = sample(sections)
        self.assertTrue(self.validate(valid, sections)["valid"])
        for name in ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                     "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
                     "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed"):
            changed = copy.deepcopy(valid)
            changed["launches"][0]["metrics"][name]["available"] = False
            self.assertFalse(self.validate(changed, sections)["valid"])

    def test_float_roofline_requires_work_and_memory_ceiling_evidence(self):
        for precision in ("overview", "half", "single", "double"):
            sections = [profile.ROOFLINE_SECTIONS[precision]]
            summary = sample(sections)
            self.assertTrue(self.validate(summary, sections)["valid"])
            del summary["launches"][0]["metrics"]["dram__bytes.sum.peak_sustained"]
            self.assertFalse(self.validate(summary, sections)["valid"])

    def test_tensor_roofline_requires_matching_precision_work_and_peak(self):
        sections = [profile.ROOFLINE_SECTIONS["tensor"]]
        summary = sample(sections)
        self.assertFalse(self.validate(summary, sections)["valid"])
        metrics = summary["launches"][0]["metrics"]
        stem = "sm__ops_path_tensor_src_fp16_dst_fp32"
        metrics[stem + ".sum.per_cycle_elapsed"] = {"value": 0., "available": True}
        metrics["sm__ops_path_tensor_src_int8.sum.peak_sustained"] = {"value": 100., "available": True}
        self.assertFalse(self.validate(summary, sections)["valid"])
        metrics[stem + ".sum.peak_sustained"] = {"value": 100., "available": True}
        self.assertTrue(self.validate(summary, sections)["valid"])
        metrics[stem + ".sum.per_cycle_elapsed"]["value"] = float("nan")
        self.assertFalse(self.validate(summary, sections)["valid"])

    def test_unknown_sections_do_not_silently_pass(self):
        self.assertFalse(self.validate(sample([]), ["UnknownSection"])["valid"])


if __name__ == "__main__":
    unittest.main()
