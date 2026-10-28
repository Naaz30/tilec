"""Metal Shading Language backend (Tier 1: scalars and 1-D tiles).

Mapping:
    program instance      -> one threadgroup
    num_warps             -> SIMD-groups per threadgroup (32 threads each)
    tile value            -> a per-thread register array, placed by its layout
    tl.program_id(axis)   -> threadgroup_position_in_grid
    reductions            -> per-thread partials, simd_sum/max/min, then
                             threadgroup memory across SIMD-groups

Runtime scalar arguments are passed as 1-element buffers, which keeps the
calling convention identical for every launcher.
"""

from __future__ import annotations

import math

from . import ir
from . import language as tl
from .layout import layout_for

SIMD_WIDTH = 32


class UnsupportedError(NotImplementedError):
    pass


def _ctype(dt: tl.dtype) -> str:
    if dt.msl is None:
        raise UnsupportedError(f"{dt!r} is not supported on Apple GPUs")
    return dt.msl


def _literal(value, dt: tl.dtype) -> str:
    if dt.is_bool:
        return "true" if value else "false"
    if dt.is_int:
        suffix = "l" if dt.bits == 64 else ""
        return f"{_ctype(dt)}({int(value)}{suffix})"
    if math.isnan(value):
        lit = "NAN"
    elif math.isinf(value):
        lit = "INFINITY" if value > 0 else "(-INFINITY)"
    else:
        lit = f"{float(value)!r}f"
    return f"{_ctype(dt)}({lit})"


_BIN_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "/",
           "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>"}
_CMP_OP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
_PRECISE = {"exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sin", "cos"}


class MSLCodegen:
    def __init__(self, func: ir.Function, num_warps: int = 4, annotate: bool = False):
        if num_warps <= 0 or num_warps & (num_warps - 1):
            raise UnsupportedError("num_warps must be a power of two")
        self.func = func
        self.num_warps = num_warps
        self.annotate = annotate    
        self.threads = num_warps * SIMD_WIDTH
        self.body: list[str] = []
        self.shared: list[str] = []         
        self.ptr_base: dict[int, str] = {}  
        self.off_type: dict[int, str] = {}  
        self.n_reduce = 0

    def layout(self, v: ir.Value):
        return layout_for(v.type.shape, self.threads)

    def name(self, v: ir.Value) -> str:
        return f"v{v.id}"

    def ref(self, v: ir.Value, r="r") -> str:
        """Expression for operand ``v`` inside a loop over register slot ``r``."""
        if v.type.is_scalar:
            return self.name(v)
        if v.type.numel == 1:            
            return f"{self.name(v)}[0]"
        return f"{self.name(v)}[{r}]"

    def elem_ctype(self, t: ir.Type) -> str:
        return _ctype(t.elem)

    def emit(self, line: str, indent=1):
        self.body.append("    " * indent + line)

    def define(self, v: ir.Value, ctype: str, expr_fn):
        """Define ``v`` from ``expr_fn(r)``: a scalar, or a loop over its registers."""
        if v.type.is_scalar:
            self.emit(f"{ctype} {self.name(v)} = {expr_fn('0')};")
            return
        lay = self.layout(v)
        self.emit(f"{ctype} {self.name(v)}[{lay.regs}];")
        self.emit(f"for (uint r = 0; r < {lay.regs}u; ++r) "
                  f"{self.name(v)}[r] = {expr_fn('r')};")

    def generate(self) -> str:
        params = []
        buffer_index = 0
        for pname, v in self.func.params:
            if v.type.is_ptr:
                ctype = _ctype(v.type.elem.elem)
                params.append(f"device {ctype}* {pname} [[buffer({buffer_index})]]")
                self.ptr_base[v.id] = pname
                self.off_type[v.id] = "int"
                self.emit(f"int {self.name(v)} = 0;  // {pname}")
            else:
                ctype = self.elem_ctype(v.type)
                params.append(
                    f"device const {ctype}* {pname}_arg [[buffer({buffer_index})]]")
                self.emit(f"const {ctype} {self.name(v)} = {pname}_arg[0];  // {pname}")
            buffer_index += 1
    
        params += [
            "uint3 tgpig [[threadgroup_position_in_grid]]",
            "uint3 tgpg [[threadgroups_per_grid]]",
            "uint3 tid3 [[thread_position_in_threadgroup]]",
            "uint lane [[thread_index_in_simdgroup]]",
            "uint sg [[simdgroup_index_in_threadgroup]]",
        ]
        self.body.insert(0, "    const uint tid = tid3.x;")

        last_loc = None
        for op in self.func.ops:
            handler = getattr(self, f"op_{op.opcode}", None)
            if handler is None:
                raise UnsupportedError(f"MSL backend: op '{op.opcode}' not supported yet")
            lay = self.layout(op.result) if op.result is not None else None 
            if self.annotate:
                if op.loc is not None and op.loc != last_loc:
                    self.emit("")
                    self.emit(f"// ===== line {op.loc[0]}: {op.loc[1]}")
                    last_loc = op.loc
                where = ""
                if lay is not None:
                    where = (f"   [layout: {lay.n} elems, {lay.regs} per thread"
                             + (", replicated" if lay.replicated else "") + "]")
                self.emit(f"// {op}{where}")
            handler(op)

        sig = ",\n    ".join(params)
        cs = ", ".join(f"{k}={v!r}" for k, v in self.func.constexprs.items())
        header = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            f"// generated by tilec  |  num_warps={self.num_warps} "
            f"({self.threads} threads/threadgroup)" + (f"  |  {cs}" if cs else ""),
            f"kernel void {self.func.name}(",
            f"    {sig})",
            "{",
        ]
        shared = ["    " + s for s in self.shared]
        return "\n".join(header + shared + self.body + ["}"]) + "\n"

    
    def op_program_id(self, op):
        axis = "xyz"[op.attrs["axis"]]
        self.emit(f"const int {self.name(op.result)} = int(tgpig.{axis});")

    def op_num_programs(self, op):
        axis = "xyz"[op.attrs["axis"]]
        self.emit(f"const int {self.name(op.result)} = int(tgpg.{axis});")

    def op_constant(self, op):
        t = op.result.type
        self.emit(f"const {self.elem_ctype(t)} {self.name(op.result)} = "
                  f"{_literal(op.attrs['value'], t.elem)};")

    def op_arange(self, op):
        lay = self.layout(op.result)
        start = op.attrs["start"]
        self.define(op.result, "int",
                    lambda r: f"{start} + int{lay.element(r, 'tid')}")

    def op_broadcast(self, op):
        src = op.operands[0]
        if not (src.type.is_scalar or src.type.numel == 1):
            raise UnsupportedError("broadcasting between non-trivial tiles needs Tier 2 layouts")
        if op.result.type.is_ptr:
            self.ptr_base[op.result.id] = self.ptr_base[src.id]
            self.off_type[op.result.id] = self.off_type[src.id]
            self.define(op.result, self.off_type[src.id], lambda r: self.ref(src, r))
        else:
            self.define(op.result, self.elem_ctype(op.result.type), lambda r: self.ref(src, r))

    def op_expand_dims(self, op):
        src = op.operands[0]
        if not src.type.is_scalar:
            raise UnsupportedError("expand_dims on tiles needs Tier 2 layouts")
        self.op_broadcast(op)

    def op_cast(self, op):
        src = op.operands[0]
        ct = self.elem_ctype(op.result.type)
        self.define(op.result, ct, lambda r: f"static_cast<{ct}>({self.ref(src, r)})")

    def op_binary(self, op):
        a, b = op.operands
        kind = op.attrs["kind"]
        t = op.result.type
        ct = self.elem_ctype(t)
        if kind in ("max", "min"):
            fn = lambda r: f"{kind}({self.ref(a, r)}, {self.ref(b, r)})"
        elif kind == "mod":
            fn = (lambda r: f"fmod({self.ref(a, r)}, {self.ref(b, r)})") if t.elem.is_float \
                else (lambda r: f"({self.ref(a, r)} % {self.ref(b, r)})")
        else:
            sym = _BIN_OP[kind]
            if t.elem.is_bool and kind in ("and", "or"):
                sym = "&&" if kind == "and" else "||"
            fn = lambda r: f"{ct}({self.ref(a, r)} {sym} {self.ref(b, r)})"
        self.define(op.result, ct, fn)

    def op_compare(self, op):
        a, b = op.operands
        sym = _CMP_OP[op.attrs["kind"]]
        self.define(op.result, "bool", lambda r: f"({self.ref(a, r)} {sym} {self.ref(b, r)})")

    def op_select(self, op):
        c, x, y = op.operands
        self.define(op.result, self.elem_ctype(op.result.type),
                    lambda r: f"({self.ref(c, r)} ? {self.ref(x, r)} : {self.ref(y, r)})")

    def op_unary(self, op):
        x = op.operands[0]
        fn = op.attrs["fn"]
        t = op.result.type
        ct = self.elem_ctype(t)
        prefix = "precise::" if (fn in _PRECISE and t.elem is tl.float32) else ""
        if fn == "sigmoid":
            exp = "precise::exp" if t.elem is tl.float32 else "exp"
            expr = lambda r: f"({ct}(1) / ({ct}(1) + {exp}(-{self.ref(x, r)})))"
        else:
            expr = lambda r: f"{prefix}{fn}({self.ref(x, r)})"
        self.define(op.result, ct, expr)

    def op_addptr(self, op):
        p, off = op.operands
        off_t = "long" if (self.off_type[p.id] == "long" or off.type.elem.bits == 64) else "int"
        self.ptr_base[op.result.id] = self.ptr_base[p.id]
        self.off_type[op.result.id] = off_t
        self.define(op.result, off_t,
                    lambda r: f"{off_t}({self.ref(p, r)}) + {off_t}({self.ref(off, r)})")

    def op_load(self, op):
        ptr = op.operands[0]
        base = self.ptr_base[ptr.id]
        ct = self.elem_ctype(op.result.type)
        if op.attrs["has_mask"]:
            mask = op.operands[1]
            other = (lambda r: self.ref(op.operands[2], r)) if op.attrs["has_other"] \
                else (lambda r: f"{ct}(0)")
            expr = lambda r: (f"({self.ref(mask, r)} ? {base}[{self.ref(ptr, r)}] "
                              f": {other(r)})")
        else:
            expr = lambda r: f"{base}[{self.ref(ptr, r)}]"
        self.define(op.result, ct, expr)

    def op_store(self, op):
        ptr, val = op.operands[0], op.operands[1]
        base = self.ptr_base[ptr.id]
        if val.type.is_scalar:
            guards = ["tid == 0u"]
            r = "0"
        else:
            lay = self.layout(val)
            g = lay.owner_guard("tid")
            guards = [g] if g else []
            r = "r"
        if op.attrs["has_mask"]:
            guards.append(self.ref(op.operands[2], r))
        stmt = f"{base}[{self.ref(ptr, r)}] = {self.ref(val, r)};"
        if guards:
            stmt = f"if ({' && '.join(guards)}) {stmt}"
        if val.type.is_scalar:
            self.emit(stmt)
        else:
            self.emit(f"for (uint r = 0; r < {lay.regs}u; ++r) {stmt}")

    def op_reduce(self, op):
        x = op.operands[0]
        kind = op.attrs["kind"]
        if x.type.is_scalar or len(x.type.shape) != 1:
            raise UnsupportedError("reductions over N-D tiles need Tier 2 layouts")
        dt = x.type.elem
        if dt.is_bool:
            raise UnsupportedError("max/min over booleans is not supported yet")
        ct = _ctype(dt)
        lay = self.layout(x)
        if kind == "sum":
            ident, combine, simd = f"{ct}(0)", "{a} + {b}", "simd_sum"
        elif kind == "max":
            ident = f"{ct}(-INFINITY)" if dt.is_float else f"numeric_limits<{ct}>::lowest()"
            combine, simd = "max({a}, {b})", "simd_max"
        else:
            ident = f"{ct}(INFINITY)" if dt.is_float else f"numeric_limits<{ct}>::max()"
            combine, simd = "min({a}, {b})", "simd_min"

        k = self.n_reduce
        self.n_reduce += 1
        scratch = f"red{k}"
        nsg = self.num_warps
        self.shared.append(f"threadgroup {ct} {scratch}[{nsg}];")
        acc = f"acc{k}"
        guard = lay.owner_guard("tid")
        step = combine.format(a=acc, b=f"{self.name(x)}[r]")
        body = f"{acc} = {step};"
        if guard:
            body = f"if ({guard}) {body}"
        self.emit(f"// reduce {kind} over {x.type.shape[0]} elements")
        self.emit(f"{ct} {acc} = {ident};")
        self.emit(f"for (uint r = 0; r < {lay.regs}u; ++r) {body}")
        self.emit(f"{acc} = {simd}({acc});")
        self.emit(f"if (lane == 0u) {scratch}[sg] = {acc};")
        self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.emit(f"{acc} = {scratch}[0];")
        self.emit(f"for (uint w = 1; w < {nsg}u; ++w) "
                  f"{acc} = {combine.format(a=acc, b=f'{scratch}[w]')};")
        self.emit(f"const {ct} {self.name(op.result)} = {acc};")


def generate_msl(func: ir.Function, num_warps: int = 4, annotate: bool = False) -> str:
    return MSLCodegen(func, num_warps, annotate).generate()
