"""Unit tests for get_gpu_inference_dtype (AMD RDNA2 support).

Pins the behavior the RX 6800 XT fix depends on: bf16 where the hardware
reports support, float32 where it doesn't (gfx1030 has no native bf16),
plus the VOICEBOX_DTYPE escape hatch.

Usage:
    python -m pytest backend/tests/test_gpu_inference_dtype.py -v
"""

import os
from unittest.mock import patch

import torch

from backend.backends.base import get_gpu_inference_dtype


class TestGetGpuInferenceDtype:
    def test_cpu_is_always_fp32(self):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            assert get_gpu_inference_dtype("cpu") == torch.float32

    def test_cuda_with_bf16_support(self):
        """Ampere+ NVIDIA / RDNA3+ ROCm keep the low-VRAM path."""
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            assert get_gpu_inference_dtype("cuda") == torch.bfloat16

    def test_cuda_without_bf16_support(self):
        """RDNA2 (RX 6800 XT, gfx1030) falls back to fp32."""
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            assert get_gpu_inference_dtype("cuda") == torch.float32

    def test_missing_bf16_symbol_falls_back(self):
        """Older torch builds without is_bf16_supported must not crash."""

        def _boom():
            raise AttributeError("no is_bf16_supported")

        with patch("torch.cuda.is_bf16_supported", side_effect=_boom):
            assert get_gpu_inference_dtype("cuda") == torch.float32

    def test_dtype_override_wins(self):
        with patch.dict(os.environ, {"VOICEBOX_DTYPE": "fp16"}):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                assert get_gpu_inference_dtype("cuda") == torch.float16

    def test_unknown_device_is_fp32(self):
        assert get_gpu_inference_dtype("mps") == torch.float32
