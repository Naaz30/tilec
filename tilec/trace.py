"""Tracing: run a kernel and print what every stage of the compiler does.

    kernel.trace[grid](args..., program=2)       # in code
    TILEC_TRACE=1 python my_script.py            # for every launch in a script
    TILEC_TRACE_PROGRAM=2 ...                    # which program instance to follow

Stages printed (each header names the file doing the work):

    1  jit.py          arguments sorted into runtime / constexpr / options, typed
    2  frontend.py     each source line → its AST → the IR ops it produced
    3  ir.py           the complete IR, every op tagged with its source line
    4  layout.py +     every op's layout decision and the Metal code it became
       codegen_msl.py  (always shown, even when running on the interpreter)
    5  interpreter.py  every op's actual values for one program instance, or
       runtime_mps.py  how the GPU launch is configured (mps backend)
"""

from __future__ import annotations

import ast
import sys

import numpy as np

from . import interpreter
from .frontend import build_ir

_RULE = "═" * 78
_MAX_ELEMS = 8

_OP_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
    ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=", ast.USub: "-", ast.UAdd: "+",
    ast.Not: "not", ast.Invert: "~", ast.And: "and", ast.Or: "or",
}


def _header(out, number, files, title):
    print(f"\n{_RULE}\n STAGE {number} · {files} · {title}\n{_RULE}", file=out)



def brief_ast(n) -> str:
    """A one-line view of an AST node that keeps its tree structure visible."""
    t = type(n)
    sym = lambda op: _OP_SYMBOLS.get(type(op), type(op).__name__)
    if t is ast.Assign:
        return f"Assign {', '.join(brief_ast(x) for x in n.targets)} = {brief_ast(n.value)}"
    if t is ast.AnnAssign:
        return f"AnnAssign {brief_ast(n.target)}: {brief_ast(n.annotation)} = {brief_ast(n.value)}"
    if t is ast.AugAssign:
        return f"AugAssign {brief_ast(n.target)} {sym(n.op)}= {brief_ast(n.value)}"
    if t is ast.Expr:
        return f"Expr {brief_ast(n.value)}"
    if t is ast.If:
        return f"If {brief_ast(n.test)}"
    if t is ast.For:
        return f"For {brief_ast(n.target)} in {brief_ast(n.iter)}"
    if t is ast.Name:
        return n.id
    if t is ast.Constant:
        return repr(n.value)
    if t is ast.Attribute:
        return f"{brief_ast(n.value)}.{n.attr}"
    if t is ast.BinOp:
        return f"BinOp({sym(n.op)}, {brief_ast(n.left)}, {brief_ast(n.right)})"
    if t is ast.Compare:
        return f"Compare({sym(n.ops[0])}, {brief_ast(n.left)}, {brief_ast(n.comparators[0])})"
    if t is ast.UnaryOp:
        return f"UnaryOp({sym(n.op)}, {brief_ast(n.operand)})"
    if t is ast.BoolOp:
        return f"BoolOp({sym(n.op)}, {', '.join(brief_ast(v) for v in n.values)})"
    if t is ast.IfExp:
        return f"IfExp({brief_ast(n.test)}, {brief_ast(n.body)}, {brief_ast(n.orelse)})"
    if t is ast.Call:
        parts = [brief_ast(a) for a in n.args]
        parts += [f"{k.arg}={brief_ast(k.value)}" for k in n.keywords]
        return f"Call({brief_ast(n.func)}{', ' if parts else ''}{', '.join(parts)})"
    if t is ast.Subscript:
        return f"Subscript({brief_ast(n.value)}, [{brief_ast(n.slice)}])"
    if t is ast.Slice:
        return ":"
    if t in (ast.Tuple, ast.List):
        return "(" + ", ".join(brief_ast(e) for e in n.elts) + ")"
    if t is ast.Return:
        return "Return"
    return t.__name__



def _fmt_array(a, limit=_MAX_ELEMS) -> str:
    a = np.asarray(a)
    if a.ndim == 0:
        v = a.item()
        return repr(round(v, 6)) if isinstance(v, float) else repr(v)
    flat = a.reshape(-1)
    if a.dtype == np.bool_:
        items = ["T" if b else "F" for b in flat[:limit]]
    elif np.issubdtype(a.dtype, np.floating):
        items = [f"{v:.4g}" for v in flat[:limit]]
    else:
        items = [str(v) for v in flat[:limit]]
    more = f", … ({flat.size} total)" if flat.size > limit else ""
    shape = f"shape {a.shape} " if a.ndim > 1 else ""
    return f"{shape}[{', '.join(items)}{more}]"


def _describe_arg(x) -> str:
    if isinstance(x, np.ndarray):
        return f"numpy {x.dtype}{list(x.shape)}"
    if type(x).__module__.startswith("torch") and hasattr(x, "device"):
        return f"torch {str(x.dtype).replace('torch.', '')}{list(x.shape)} on {x.device}"
    return repr(x)



def trace_launch(jit_fn, grid, args, kwargs, program=0, out=None):
    out = out or sys.stdout
    p = lambda *a: print(*a, file=out)

   
    runtime, constexprs, options = jit_fn._bind(args, dict(kwargs))
    num_warps = int(options.get("num_warps", 4))
    from .jit import _arg_type, CompiledKernel
    arg_types = {k: _arg_type(v) for k, v in runtime.items()}
    grid = jit_fn.resolve_grid(grid, runtime, constexprs)
    backend = jit_fn.choose_backend(options, runtime)

    _header(out, 1, "jit.py", f"binding the arguments of {jit_fn.__name__}")
    params = list(jit_fn.signature.parameters)
    width = max(len(n) for n in params)
    bound = dict(runtime, **constexprs)
    for name in params:
        if name in constexprs:
            kind = "constexpr → baked into the compiled code"
        else:
            kind = f"runtime   → IR parameter of type {arg_types[name]}"
        p(f"  {name:<{width}}  {_describe_arg(bound[name]):<30} {kind}")
    p(f"\n  grid      = {grid}   ({int(np.prod(grid))} program instances)")
    p(f"  num_warps = {num_warps}   ({num_warps * 32} threads per program on the GPU)")
    p(f"  backend   = {backend}")

    
    _header(out, 2, "frontend.py", "source line → AST → IR ops (one block per statement)")

    def on_stmt(stmt, loc, new_ops, changed):
        line, text = loc
        p(f"\n  line {line} │ {text}")
        p(f"   AST    │ {brief_ast(stmt)}")
        if not new_ops:
            p("   IR     │ (nothing: everything here was known at compile time)")
        for i, op in enumerate(new_ops):
            tag = "   IR     │ " if i == 0 else "          │ "
            inner = f"    // from line {op.loc[0]}" if op.loc and op.loc != loc else ""
            p(f"{tag}{op}{inner}")
        for name, v in changed.items():
            shown = f"{v}  : {v.type}" if hasattr(v, "type") else repr(v)
            p(f"   names  │ {name} → {shown}")

    func = build_ir(jit_fn.fn, arg_types, constexprs, hook=on_stmt)
    ck = CompiledKernel(func, num_warps)

   
    _header(out, 3, "ir.py", f"complete IR ({len(func.ops)} ops) — input to both backends")
    p(func.format(with_locs=True))

   
    _header(out, 4, "layout.py + codegen_msl.py",
            "IR → Metal Shading Language (each block: source line, IR op, its code)")
    msl = None
    try:
        from .codegen_msl import generate_msl
        msl = generate_msl(func, num_warps, annotate=True)
        p(msl)
        if "replicated]" in msl:
            p(f"  note: a tile with fewer elements than the {num_warps * 32} threads is "
              "'replicated' —\n  several threads hold the same element, and only "
              "threads tid < N store, so nothing is written twice.")
    except NotImplementedError as e:
        p(f"  The Metal backend cannot compile this kernel yet:\n  {e}")
    if backend != "mps":
        p("  (shown for reference: this launch runs on the interpreter, not the GPU)")

    
    if backend == "interpreter":
        pid = tuple(program) if isinstance(program, (tuple, list)) else (int(program),)
        pid = pid + (0,) * (3 - len(pid))
        if any(pid[i] >= (tuple(grid) + (1, 1))[i] for i in range(3)):
            raise ValueError(f"program {pid} is outside the grid {grid}")
        _header(out, 5, "interpreter.py",
                f"executing — values of every op in program {pid[0] if pid[1:] == (0, 0) else pid}")
        ptr_names = [name for name, v in func.params if v.type.is_ptr]
        state = {"loc": None}

        def show_ptr(ptr):
            return f"{ptr_names[ptr.buf]} + {_fmt_array(ptr.offsets)}"

        def observer(op, op_args, value):
            if op.loc is not None and op.loc != state["loc"]:
                p(f"\n  line {op.loc[0]} │ {op.loc[1]}")
                state["loc"] = op.loc
            label = f"{op.result}" if op.result is not None else "  "
            desc = op.opcode + (f" {op.attrs['kind']}" if "kind" in op.attrs else "") \
                + (f" {op.attrs['fn']}" if "fn" in op.attrs else "")
            if op.opcode == "store":
                ptr, val = op_args[0], np.asarray(op_args[1])
                offs = np.broadcast_to(ptr.offsets, val.shape)
                if op.attrs["has_mask"]:
                    m = np.broadcast_to(op_args[2], np.shape(val))
                    offs, val = offs[m], val[m]
                res = (f"writes {_fmt_array(val)} to {ptr_names[ptr.buf]}"
                       f"[{_fmt_array(offs)}]")
            elif isinstance(value, interpreter.Ptr):
                res = show_ptr(value)
            else:
                res = _fmt_array(value)
            p(f"    {label:<5} {desc:<16} = {res}")

        jit_fn.launch(ck, grid, runtime, backend, num_warps, observer, pid)
        p(f"\n  (all {int(np.prod(grid))} programs ran; only program {pid[0]} was printed)")

    elif backend == "mps":
        _header(out, 5, "runtime_mps.py", "compiling the Metal code and launching it on the GPU")
        threads = num_warps * 32
        gx, gy, gz = (tuple(grid) + (1, 1))[:3]
        p("  torch.mps.compile_shader(<the Metal source above>)")
        p(f"  launch: threads = [{gx * threads}, {gy}, {gz}], group_size = [{threads}, 1, 1]")
        p(f"          → {gx * gy * gz} threadgroups × {threads} threads, all running at once")
        for name, v in func.params:
            if not v.type.is_ptr:
                p(f"  scalar {name} = {runtime[name]!r} passed as a 1-element buffer")
        if msl is not None:
            ck._msl = msl
        jit_fn.launch(ck, grid, runtime, backend, num_warps)
        import torch
        torch.mps.synchronize()
        p("  done (torch.mps.synchronize() returned)")

   
    _header(out, "✓", backend, "buffers after the launch")
    for name, v in func.params:
        if v.type.is_ptr:
            a = runtime[name]
            data = a.detach().cpu().numpy() if hasattr(a, "detach") else a
            p(f"  {name:<{width}} = {_fmt_array(data, limit=16)}")
    p("")
    return ck
