#!/usr/bin/env python3
"""GPU diagnostic for AMD setups (RX 6800 XT and friends).

Prints everything needed to debug Voicebox GPU inference in one shot:
platform, torch build (HIP vs CUDA), device visibility, bf16 support,
HSA override, and the dtype Voicebox will actually load Qwen models with.

Usage:
    python3 scripts/check_amd.py

Exit code is 0 when a GPU is visible to torch, 1 otherwise (CPU fallback).
No model downloads, no server start — safe to run anywhere.
"""

import os
import platform
import sys


def main() -> int:
    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"python   : {sys.version.split()[0]}")
    print(f"HSA_OVERRIDE_GFX_VERSION : {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '(unset)')}")
    print(f"VOICEBOX_DTYPE           : {os.environ.get('VOICEBOX_DTYPE', '(unset)')}")
    print(f"VOICEBOX_FORCE_CPU       : {os.environ.get('VOICEBOX_FORCE_CPU', '(unset)')}")
    print(f"/dev/kfd present         : {os.path.exists('/dev/kfd')}")

    try:
        import torch
    except ImportError:
        print("torch    : NOT INSTALLED")
        return 1

    print(f"torch    : {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"hip build: {hip}")
    print(f"cuda ver : {cuda_ver}")

    avail = torch.cuda.is_available()
    print(f"cuda_available (covers HIP): {avail}")
    if not avail:
        print("result   : no GPU visible — Voicebox will run on CPU")
        print("hint     : Linux AMD needs ROCm torch, e.g.")
        print("           VOICEBOX_TORCH_INDEX=https://download.pytorch.org/whl/rocm6.3 just setup-python")
        return 1

    print(f"device   : {torch.cuda.get_device_name(0)}")
    try:
        print(f"bf16 ok  : {torch.cuda.is_bf16_supported()}")
    except Exception as e:  # old builds lack the symbol
        print(f"bf16 ok  : unknown ({e})")

    # SDPA backends matter for transformer TTS on gfx10xx.
    # getattr (not from-import) so old torch builds without these symbols
    # report "unknown" instead of crashing the diagnostic.
    try:
        from torch.backends import cuda as cuda_backends

        print(
            "sdpa     : flash="
            f"{getattr(cuda_backends, 'is_flash_sdp_enabled', lambda: '?')()} "
            "mem_efficient="
            f"{getattr(cuda_backends, 'is_mem_efficient_sdp_enabled', lambda: '?')()} "
            "math="
            f"{getattr(cuda_backends, 'is_math_sdp_enabled', lambda: '?')()}"
        )
    except Exception as e:
        print(f"sdpa     : unknown ({e})")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from backend.backends.base import get_gpu_inference_dtype

        print(f"qwen dtype on cuda: {get_gpu_inference_dtype('cuda')}")
    except Exception as e:
        print(f"qwen dtype: could not resolve ({e})")

    print("result   : GPU visible, ready for Qwen inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
