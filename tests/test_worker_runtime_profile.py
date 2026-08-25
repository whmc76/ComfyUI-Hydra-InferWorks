import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = spec_from_file_location("hydra_inferworks_worker_contract", _ROOT / "worker.py")
_MODULE = module_from_spec(_SPEC)
_stdout = sys.stdout
try:
    _SPEC.loader.exec_module(_MODULE)
finally:
    sys.stdout = _stdout


def _linear(dtype):
    return torch.nn.Linear(2, 2).to(dtype=dtype)


class WorkerRuntimeProfileTests(unittest.TestCase):
    def test_quality_bf16_reports_actual_eager_profile(self):
        model = SimpleNamespace(
            device="cuda:0",
            use_bf16=True,
            use_torch_compile=False,
            use_cuda_kernel=False,
            gpt=_linear(torch.bfloat16),
            s2mel=SimpleNamespace(models={"cfm": SimpleNamespace(estimator=object())}, parameters=lambda: iter([torch.nn.Parameter(torch.ones(1, dtype=torch.float32))])),
            bigvgan=_linear(torch.float32),
        )
        profile = _MODULE.runtime_profile(
            model,
            {"optimization_profile": "quality_bf16"},
            synthesis_completed=True,
        )
        self.assertEqual(profile["precision"], "mixed_bf16_fp32")
        self.assertEqual(profile["quality_admission"], "production")
        self.assertFalse(any(profile["acceleration"].values()))
        _MODULE.require_production_profile_truth(profile)
        forged = dict(profile)
        forged["precision"] = "bf16"
        with self.assertRaisesRegex(RuntimeError, "runtime_attestation_failed"):
            _MODULE.require_production_profile_truth(forged)

    def test_maximum_profile_reports_mixed_fp16_gpt_stage(self):
        engine = SimpleNamespace(model=_linear(torch.float16))
        gpt = _linear(torch.bfloat16)
        gpt.accel_engine = engine
        estimator = SimpleNamespace(_orig_mod=object())
        model = SimpleNamespace(
            device="cuda:0",
            use_bf16=True,
            use_torch_compile=True,
            use_cuda_kernel=True,
            gpt=gpt,
            s2mel=SimpleNamespace(models={"cfm": SimpleNamespace(estimator=estimator)}, parameters=lambda: iter([torch.nn.Parameter(torch.ones(1, dtype=torch.float32))])),
            bigvgan=_linear(torch.float32),
        )
        profile = _MODULE.runtime_profile(
            model,
            {"optimization_profile": "maximum_bf16"},
            synthesis_completed=True,
        )
        self.assertEqual(profile["precision"], "mixed_bf16_fp16_fp32")
        self.assertEqual(profile["stage_precisions"]["gpt"], "fp16")
        self.assertEqual(profile["acceleration_state"]["torch_compile"], "executed")
        self.assertEqual(profile["quality_admission"], "candidate")


if __name__ == "__main__":
    unittest.main()
