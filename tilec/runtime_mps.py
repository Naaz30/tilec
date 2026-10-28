"""Launch generated MSL on Apple GPUs through PyTorch's MPS backend.

Uses ``torch.mps.compile_shader`` (PyTorch >= 2.5), which compiles MSL source
and lets Python call its kernels on MPS tensors with zero copies.
"""

from __future__ import annotations

from . import language as tl

_LIBRARIES: dict[str, object] = {}


def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("the Metal backend needs PyTorch with MPS support") from e
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch.backends.mps is not available on this machine")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("torch.mps.compile_shader needs PyTorch 2.5 or newer")
    return torch


def launch(source: str, kernel_name: str, params, args, grid, num_warps: int):
    """params: list of (name, ir.Value) in order; args: matching Python values."""
    torch = _torch()
    lib = _LIBRARIES.get(source)
    if lib is None:
        lib = torch.mps.compile_shader(source)
        _LIBRARIES[source] = lib
    kernel = getattr(lib, kernel_name)

    packed = []
    for (name, v), a in zip(params, args):
        if v.type.is_ptr:
            if not (torch.is_tensor(a) and a.device.type == "mps"):
                raise TypeError(f"argument '{name}' must be an MPS tensor")
            if not a.is_contiguous():
                raise ValueError(f"argument '{name}' must be contiguous")
            packed.append(a)
        else:
            tdt = _TORCH_DTYPES[v.type.elem](torch)
            packed.append(torch.tensor([a], dtype=tdt, device="mps"))

    threads = num_warps * 32
    gx, gy, gz = (tuple(grid) + (1, 1))[:3]
    kernel(*packed, threads=[gx * threads, gy, gz], group_size=[threads, 1, 1])


_TORCH_DTYPES = {
    tl.int1: lambda t: t.bool,
    tl.int8: lambda t: t.int8,
    tl.int16: lambda t: t.int16,
    tl.int32: lambda t: t.int32,
    tl.int64: lambda t: t.int64,
    tl.float16: lambda t: t.float16,
    tl.bfloat16: lambda t: t.bfloat16,
    tl.float32: lambda t: t.float32,
}


def torch_dtype_to_tl(dt):
    name = str(dt).replace("torch.", "")
    mapping = {"bool": tl.int1, "int8": tl.int8, "int16": tl.int16, "int32": tl.int32,
               "int64": tl.int64, "float16": tl.float16, "bfloat16": tl.bfloat16,
               "float32": tl.float32, "float64": tl.float64}
    if name not in mapping:
        raise TypeError(f"unsupported tensor dtype {dt}")
    return mapping[name]
