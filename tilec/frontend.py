from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap

import numpy as np

from . import ir
from . import language as tl


class CompileError(Exception):
    pass


_MAX_UNROLL = 256

_BIN_KINDS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.BitAnd: "and",
    ast.BitOr: "or", ast.BitXor: "xor", ast.LShift: "shl", ast.RShift: "shr",
}
_PY_BIN = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "div": operator.truediv, "floordiv": operator.floordiv,
    "mod": operator.mod, "and": operator.and_, "or": operator.or_,
    "xor": operator.xor, "shl": operator.lshift, "shr": operator.rshift,
}
_CMP_KINDS = {
    ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
    ast.Eq: "eq", ast.NotEq: "ne",
}
_PY_CMP = {
    "lt": operator.lt, "le": operator.le, "gt": operator.gt,
    "ge": operator.ge, "eq": operator.eq, "ne": operator.ne,
}
_UNARY_MATH = {"exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sin", "cos",
               "floor", "ceil", "sigmoid"}
_FLOAT_ONLY_MATH = _UNARY_MATH 


def _is_value(x):
    return isinstance(x, ir.Value)


def _is_const_number(x):
    return isinstance(x, (bool, int, float, np.integer, np.floating, np.bool_))


def promote(a: tl.dtype, b: tl.dtype) -> tl.dtype:
    """Binary-op result dtype for two tensor/scalar Values (Triton-like rules)."""
    if a is b:
        return a
    if a.is_float or b.is_float:
        if a.is_float and b.is_float:
            if a.bits != b.bits:
                return a if a.bits > b.bits else b
            return tl.float32         
        return a if a.is_float else b
    if a.is_bool:
        return b
    if b.is_bool:
        return a
    return a if a.bits >= b.bits else b


class FrontEnd:
    def __init__(self, fn, arg_types: dict, constexprs: dict):
        self.fn = fn
        self.arg_types = arg_types          
        self.constexprs = constexprs        
        self.func = ir.Function(fn.__name__)
        self.func.constexprs = dict(constexprs)
        self.env: dict = {}
        self.globals = dict(fn.__globals__)
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                self.globals[name] = cell.cell_contents
        self.src_lines = []
        self.first_line = 1
        self.hook = None

    
    def build(self) -> ir.Function:
        try:
            src, self.first_line = inspect.getsourcelines(self.fn)
        except OSError as e:
            raise CompileError(f"cannot read source of {self.fn.__name__}: {e}")
        src = textwrap.dedent("".join(src))
        self.src_lines = src.splitlines()
        tree = ast.parse(src)
        fdef = tree.body[0]
        if not isinstance(fdef, ast.FunctionDef):
            raise CompileError("@jit must decorate a plain function")

        for name in [a.arg for a in fdef.args.args]:
            if name in self.constexprs:
                self.env[name] = self.constexprs[name]
            else:
                self.env[name] = self.func.add_param(name, self.arg_types[name])

        body = fdef.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]                                   
        self.visit_block(body, top_level=True)
        return self.func

    def loc(self, node):
        """(line number in the user's file, source text) for an AST node."""
        line = getattr(node, "lineno", None)
        if line is None:
            return None
        text = self.src_lines[line - 1].strip() if line - 1 < len(self.src_lines) else ""
        return (line + self.first_line - 1, text)

    def error(self, node, msg):
        line = getattr(node, "lineno", None)
        where = ""
        if line is not None:
            src = self.src_lines[line - 1].strip() if line - 1 < len(self.src_lines) else ""
            where = f"\n  at line {line + self.first_line - 1}: {src}"
        return CompileError(f"{msg}{where}")

    
    def visit_block(self, stmts, top_level=False):
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    raise self.error(stmt, "kernels cannot return values")
                if not (top_level and i == len(stmts) - 1):
                    raise self.error(stmt, "early return is not supported yet")
                return
            self.func.cur_loc = self.loc(stmt)
            if top_level and self.hook is not None:
                n_before = len(self.func.ops)
                env_before = dict(self.env)
                self.visit_stmt(stmt)
                changed = {k: v for k, v in self.env.items()
                           if k not in env_before or env_before[k] is not v}
                self.hook(stmt, self.loc(stmt), self.func.ops[n_before:], changed)
            else:
                self.visit_stmt(stmt)

    def visit_stmt(self, node):
        if isinstance(node, ast.Assign):
            value = self.eval(node.value)
            for target in node.targets:
                self.assign(target, value)
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                raise self.error(node, "annotation without a value")
            self.assign(node.target, self.eval(node.value))
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise self.error(node, "only `name op= value` is supported")
            kind = _BIN_KINDS.get(type(node.op))
            if kind is None:
                raise self.error(node, f"unsupported operator {type(node.op).__name__}")
            cur = self.lookup(node.target.id, node)
            self.env[node.target.id] = self.binary(kind, cur, self.eval(node.value), node)
        elif isinstance(node, ast.Expr):
            self.eval(node.value)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.If):
            test = self.eval(node.test)
            if _is_value(test):
                raise self.error(node, "`if` on a runtime value is not supported yet "
                                       "(Tier 1 supports `if` on constexpr values; "
                                       "use tl.where for elementwise choices)")
            self.visit_block(node.body if test else node.orelse)
        elif isinstance(node, ast.For):
            self.visit_for(node)
        elif isinstance(node, ast.Assert):
            test = self.eval(node.test)
            if _is_value(test):
                raise self.error(node, "runtime assert is not supported yet; "
                                       "use tl.static_assert for constexpr checks")
            if not test:
                msg = self.eval(node.msg) if node.msg is not None else "assertion failed"
                raise self.error(node, f"static assertion failed: {msg}")
        else:
            raise self.error(node, f"unsupported statement: {type(node).__name__}")

    def visit_for(self, node):
        if node.orelse:
            raise self.error(node, "for-else is not supported")
        it = node.iter
        ok = (isinstance(it, ast.Call)
              and (self._resolves_to(it.func, tl.BUILTINS["static_range"])
                   or self._resolves_to(it.func, builtins.range)))
        if not ok:
            raise self.error(node, "only `for i in tl.static_range(...)` or "
                                   "`range(...)` with constant bounds is supported")
        args = [self.eval(a) for a in it.args]
        if any(_is_value(a) for a in args):
            raise self.error(node, "loops with runtime bounds are not supported yet "
                                   "(Tier 1 unrolls loops with constant bounds)")
        rng = builtins.range(*[int(a) for a in args])
        if len(rng) > _MAX_UNROLL:
            raise self.error(node, f"loop of {len(rng)} iterations is too long to unroll "
                                   f"(limit {_MAX_UNROLL}); runtime loops come in Tier 2")
        for i in rng:
            self.assign(node.target, i)
            self.visit_block(node.body)

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
                raise self.error(target, "tuple unpacking needs a tuple of matching length")
            for t, v in zip(target.elts, value):
                self.assign(t, v)
        else:
            raise self.error(target, "only assignment to names is supported")

    def _resolves_to(self, func_node, obj):
        try:
            return self.eval(func_node) is obj
        except CompileError:
            return False

    def lookup(self, name, node):
        if name in self.env:
            return self.env[name]
        if name in self.globals:
            return self.globals[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise self.error(node, f"name '{name}' is not defined")

    def eval(self, node):
        method = getattr(self, f"eval_{type(node).__name__}", None)
        if method is None:
            raise self.error(node, f"unsupported expression: {type(node).__name__}")
        return method(node)

    def eval_Constant(self, node):
        return node.value

    def eval_Name(self, node):
        return self.lookup(node.id, node)

    def eval_Tuple(self, node):
        return tuple(self.eval(e) for e in node.elts)

    eval_List = eval_Tuple

    def eval_Attribute(self, node):
        base = self.eval(node.value)
        if _is_value(base):
            if node.attr == "dtype":
                if base.type.is_ptr:
                    return base.type.elem
                return base.type.elem
            if node.attr == "shape":
                return base.type.shape
            raise self.error(node, f"tensors have no attribute '{node.attr}' "
                                   "(only .dtype, .shape and .to() are supported)")
        if isinstance(base, tl.constexpr):
            base = base.value
        try:
            return getattr(base, node.attr)
        except AttributeError:
            raise self.error(node, f"{base!r} has no attribute '{node.attr}'")

    def eval_BinOp(self, node):
        kind = _BIN_KINDS.get(type(node.op))
        if kind is None:
            raise self.error(node, f"unsupported operator {type(node.op).__name__}")
        return self.binary(kind, self.eval(node.left), self.eval(node.right), node)

    def eval_UnaryOp(self, node):
        x = self.eval(node.operand)
        if not _is_value(x):
            ops = {ast.USub: operator.neg, ast.UAdd: operator.pos,
                   ast.Not: operator.not_, ast.Invert: operator.invert}
            return ops[type(node.op)](x)
        if isinstance(node.op, ast.UAdd):
            return x
        if isinstance(node.op, ast.USub):
            return self.binary("sub", 0, x, node)
        if isinstance(node.op, ast.Not):
            return self.compare("eq", x, 0, node)
        if isinstance(node.op, ast.Invert):
            if x.type.is_ptr or x.type.elem.is_float:
                raise self.error(node, "~ needs an integer or boolean operand")
            if x.type.elem.is_bool:
                return self.compare("eq", x, False, node)
            return self.binary("xor", x, -1, node)
        raise self.error(node, "unsupported unary operator")

    def eval_Compare(self, node):
        if len(node.ops) != 1:
            raise self.error(node, "chained comparisons are not supported")
        kind = _CMP_KINDS.get(type(node.ops[0]))
        if kind is None:
            raise self.error(node, f"unsupported comparison {type(node.ops[0]).__name__}")
        return self.compare(kind, self.eval(node.left), self.eval(node.comparators[0]), node)

    def eval_BoolOp(self, node):
        vals = [self.eval(v) for v in node.values]
        if not any(_is_value(v) for v in vals):
            result = vals[0]
            for v in vals[1:]:
                result = (result and v) if isinstance(node.op, ast.And) else (result or v)
            return result
        kind = "and" if isinstance(node.op, ast.And) else "or"
        result = self.to_bool(vals[0], node)
        for v in vals[1:]:
            result = self.binary(kind, result, self.to_bool(v, node), node)
        return result

    def eval_IfExp(self, node):
        test = self.eval(node.test)
        if not _is_value(test):
            return self.eval(node.body if test else node.orelse)
        return self.where(test, self.eval(node.body), self.eval(node.orelse), node)

    def eval_Subscript(self, node):
        base = self.eval(node.value)
        if not _is_value(base):
            idx = self.eval(node.slice)
            return base[idx]
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        out = base
        axis = 0
        for item in items:
            if isinstance(item, ast.Constant) and item.value is None:
                out = self.expand_dims(out, axis, node)
                axis += 1
            elif (isinstance(item, ast.Slice) and item.lower is None
                  and item.upper is None and item.step is None):
                axis += 1
            else:
                raise self.error(node, "only `:` and `None` are supported when indexing tensors")
        if axis != len(out.type.shape):
            raise self.error(node, f"indexing uses {axis} dims but the tensor has "
                                   f"{len(out.type.shape)}")
        return out

    def eval_Call(self, node):
        # method call: x.to(dtype)
        if isinstance(node.func, ast.Attribute):
            base = self.eval(node.func.value)
            if _is_value(base):
                if node.func.attr == "to":
                    args, kwargs = self._call_args(node)
                    dt = args[0] if args else kwargs.get("dtype")
                    return self.cast(base, self._dtype_arg(dt, node), node)
                raise self.error(node, f"unsupported tensor method .{node.func.attr}()")
        fn = self.eval(node.func)
        args, kwargs = self._call_args(node)
        if isinstance(fn, tl.Builtin):
            handler = getattr(self, f"b_{fn.name}", None)
            if handler is None:
                raise self.error(node, f"tl.{fn.name} is not implemented yet")
            return handler(node, *args, **kwargs)
        if callable(fn) and not any(_is_value(a) for a in list(args) + list(kwargs.values())):
            if getattr(fn, "__module__", None) == "builtins" or fn in (int, float, bool):
                return fn(*args, **kwargs)
        if hasattr(fn, "__tilec_jit__"):
            raise self.error(node, "calling another @jit function is not supported yet")
        raise self.error(node, f"cannot call {fn!r} inside a kernel")

    def _call_args(self, node):
        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise self.error(node, "*args is not supported")
            args.append(self.eval(a))
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise self.error(node, "**kwargs is not supported")
            kwargs[kw.arg] = self.eval(kw.value)
        return args, kwargs

    def _dtype_arg(self, dt, node):
        if isinstance(dt, tl.constexpr):
            dt = dt.value
        if not isinstance(dt, tl.dtype):
            raise self.error(node, f"expected a dtype like tl.float32, got {dt!r}")
        return dt

    def _const_int(self, x, node, what):
        if isinstance(x, tl.constexpr):
            x = x.value
        if _is_value(x) or not isinstance(x, (int, np.integer)):
            raise self.error(node, f"{what} must be a compile-time integer")
        return int(x)

    def _shape_arg(self, shape, node):
        if isinstance(shape, (int, np.integer)):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            raise self.error(node, "shape must be a tuple of compile-time integers")
        return tuple(self._const_int(s, node, "every shape entry") for s in shape)

    def constant(self, value, dt):
        if dt.is_bool:
            value = bool(value)
        elif dt.is_int:
            value = int(value)
        else:
            value = builtins.float(value)
        return self.func.emit("constant", [], ir.Type(dt), value=value)

    def _weak_dtype(self, const, other_dt):
        """dtype for a Python number used next to a typed Value."""
        if isinstance(const, (bool, np.bool_)):
            return tl.int1 if other_dt is None or other_dt.is_bool else other_dt
        if isinstance(const, (int, np.integer)):
            if other_dt is None or other_dt.is_bool:
                return tl.int32
            return other_dt
        return other_dt if (other_dt is not None and other_dt.is_float) else tl.float32

    def to_value(self, x, node, like=None):
        if _is_value(x):
            return x
        if isinstance(x, tl.constexpr):
            x = x.value
        if not _is_const_number(x):
            raise self.error(node, f"expected a number or tensor, got {x!r}")
        other = None
        if like is not None and not like.type.is_ptr:
            other = like.type.elem
        return self.constant(x, self._weak_dtype(x, other))

    def broadcast_to(self, v, shape, node):
        shape = tuple(shape)
        if v.type.shape == shape:
            return v
        try:
            if np.broadcast_shapes(v.type.shape, shape) != shape:
                raise ValueError
        except ValueError:
            raise self.error(node, f"cannot broadcast {v.type.shape} to {shape}")
        return self.func.emit("broadcast", [v], v.type.with_shape(shape))

    def broadcast_pair(self, a, b, node):
        try:
            shape = tuple(np.broadcast_shapes(a.type.shape, b.type.shape))
        except ValueError:
            raise self.error(node, f"shapes {a.type.shape} and {b.type.shape} "
                                   "are not broadcast-compatible")
        return self.broadcast_to(a, shape, node), self.broadcast_to(b, shape, node)

    def cast(self, v, dt, node):
        v = self.to_value(v, node)
        if v.type.is_ptr:
            raise self.error(node, "cannot cast a pointer")
        if v.type.elem is dt:
            return v
        return self.func.emit("cast", [v], v.type.with_elem(dt))

    def to_bool(self, v, node):
        v = self.to_value(v, node)
        if not v.type.is_ptr and v.type.elem.is_bool:
            return v
        return self.compare("ne", v, 0, node)

    def expand_dims(self, v, axis, node):
        shape = list(v.type.shape)
        if not 0 <= axis <= len(shape):
            raise self.error(node, f"expand_dims axis {axis} out of range")
        shape.insert(axis, 1)
        return self.func.emit("expand_dims", [v], v.type.with_shape(shape), axis=axis)

    def binary(self, kind, a, b, node):
        if isinstance(a, tl.constexpr):
            a = a.value
        if isinstance(b, tl.constexpr):
            b = b.value
        if not _is_value(a) and not _is_value(b):
            try:
                return _PY_BIN[kind](a, b)
            except Exception as e:
                raise self.error(node, f"compile-time arithmetic failed: {e}")

        # pointer arithmetic
        a_ptr = _is_value(a) and a.type.is_ptr
        b_ptr = _is_value(b) and b.type.is_ptr
        if a_ptr or b_ptr:
            if a_ptr and b_ptr:
                raise self.error(node, "cannot combine two pointers")
            if kind == "add":
                ptr, off = (a, b) if a_ptr else (b, a)
            elif kind == "sub" and a_ptr:
                ptr, off = a, self.binary("sub", 0, b, node)
            else:
                raise self.error(node, f"pointers only support + and - (got {kind})")
            off = self.to_value(off, node)
            if off.type.is_ptr or not off.type.elem.is_int:
                raise self.error(node, "pointer offsets must be integers")
            ptr, off = self.broadcast_pair(ptr, off, node)
            return self.func.emit("addptr", [ptr, off], ptr.type)

        a = self.to_value(a, node, like=b if _is_value(b) else None)
        b = self.to_value(b, node, like=a)
        dt = promote(a.type.elem, b.type.elem)

        if kind == "div" and not dt.is_float:
            dt = tl.float32                       # int / int -> f32, like Triton
        if kind in ("and", "or", "xor", "shl", "shr") and dt.is_float:
            raise self.error(node, f"bitwise '{kind}' needs integer operands")
        if kind == "floordiv" and dt.is_float:
            raise self.error(node, "// on floats is not supported; use tl.floor(a / b)")

        a, b = self.cast(a, dt, node), self.cast(b, dt, node)
        a, b = self.broadcast_pair(a, b, node)
        return self.func.emit("binary", [a, b], a.type, kind=kind)

    def compare(self, kind, a, b, node):
        if isinstance(a, tl.constexpr):
            a = a.value
        if isinstance(b, tl.constexpr):
            b = b.value
        if not _is_value(a) and not _is_value(b):
            return _PY_CMP[kind](a, b)
        a = self.to_value(a, node, like=b if _is_value(b) else None)
        b = self.to_value(b, node, like=a)
        if a.type.is_ptr or b.type.is_ptr:
            raise self.error(node, "comparing pointers is not supported yet")
        dt = promote(a.type.elem, b.type.elem)
        a, b = self.cast(a, dt, node), self.cast(b, dt, node)
        a, b = self.broadcast_pair(a, b, node)
        return self.func.emit("compare", [a, b], a.type.with_elem(tl.int1), kind=kind)

    def where(self, cond, x, y, node):
        cond = self.to_bool(cond, node)
        if not _is_value(x) and not _is_value(y):
            x = self.to_value(x, node)
        x = self.to_value(x, node, like=y if _is_value(y) else None)
        y = self.to_value(y, node, like=x)
        dt = promote(x.type.elem, y.type.elem)
        x, y = self.cast(x, dt, node), self.cast(y, dt, node)
        shape = tuple(np.broadcast_shapes(cond.type.shape, x.type.shape, y.type.shape))
        cond = self.broadcast_to(cond, shape, node)
        x, y = self.broadcast_to(x, shape, node), self.broadcast_to(y, shape, node)
        return self.func.emit("select", [cond, x, y], x.type)

    def b_program_id(self, node, axis=0):
        axis = self._const_int(axis, node, "axis")
        if axis not in (0, 1, 2):
            raise self.error(node, "program_id axis must be 0, 1 or 2")
        return self.func.emit("program_id", [], ir.Type(tl.int32), axis=axis)

    def b_num_programs(self, node, axis=0):
        axis = self._const_int(axis, node, "axis")
        if axis not in (0, 1, 2):
            raise self.error(node, "num_programs axis must be 0, 1 or 2")
        return self.func.emit("num_programs", [], ir.Type(tl.int32), axis=axis)

    def b_arange(self, node, start, end):
        start = self._const_int(start, node, "arange start")
        end = self._const_int(end, node, "arange end")
        n = end - start
        if n <= 0 or (n & (n - 1)) != 0:
            raise self.error(node, f"arange size must be a positive power of two (got {n})")
        return self.func.emit("arange", [], ir.Type(tl.int32, (n,)), start=start, end=end)

    def b_full(self, node, shape, value, dtype):
        shape = self._shape_arg(shape, node)
        dt = self._dtype_arg(dtype, node)
        v = self.cast(self.to_value(value, node), dt, node) if _is_value(value) \
            else self.constant(value.value if isinstance(value, tl.constexpr) else value, dt)
        return self.broadcast_to(v, shape, node)

    def b_zeros(self, node, shape, dtype):
        return self.b_full(node, shape, 0, dtype)

    def b_cast(self, node, x, dtype):
        return self.cast(x, self._dtype_arg(dtype, node), node)

    def b_load(self, node, pointer, mask=None, other=None, **hints):
        self._check_hints(node, hints, {"cache_modifier", "eviction_policy", "volatile"})
        if not (_is_value(pointer) and pointer.type.is_ptr):
            raise self.error(node, "tl.load needs a pointer or a tensor of pointers")
        elem = pointer.type.elem.elem
        operands = [pointer]
        shape = pointer.type.shape
        if mask is not None:
            m = self.to_bool(mask, node)
            shape = tuple(np.broadcast_shapes(shape, m.type.shape))
            pointer = self.broadcast_to(pointer, shape, node)
            operands = [pointer, self.broadcast_to(m, shape, node)]
        if other is not None:
            if mask is None:
                raise self.error(node, "`other` requires `mask`")
            o = self.cast(self.to_value(other, node), elem, node) if _is_value(other) \
                else self.constant(other.value if isinstance(other, tl.constexpr) else other, elem)
            operands.append(self.broadcast_to(o, shape, node))
        return self.func.emit("load", operands, ir.Type(elem, shape),
                              has_mask=mask is not None, has_other=other is not None)

    def b_store(self, node, pointer, value, mask=None, **hints):
        self._check_hints(node, hints, {"cache_modifier", "eviction_policy"})
        if not (_is_value(pointer) and pointer.type.is_ptr):
            raise self.error(node, "tl.store needs a pointer or a tensor of pointers")
        elem = pointer.type.elem.elem
        value = self.cast(self.to_value(value, node), elem, node)
        shapes = [pointer.type.shape, value.type.shape]
        m = None
        if mask is not None:
            m = self.to_bool(mask, node)
            shapes.append(m.type.shape)
        shape = tuple(np.broadcast_shapes(*shapes))
        operands = [self.broadcast_to(pointer, shape, node), self.broadcast_to(value, shape, node)]
        if m is not None:
            operands.append(self.broadcast_to(m, shape, node))
        self.func.emit("store", operands, None, has_mask=m is not None)

    def _check_hints(self, node, hints, allowed):
        for k in hints:
            if k not in allowed:
                raise self.error(node, f"unsupported keyword argument '{k}'")

    def b_where(self, node, condition, x, y):
        return self.where(condition, x, y, node)

    def b_maximum(self, node, x, y):
        return self.binary("max", x, y, node) if (_is_value(x) or _is_value(y)) \
            else builtins.max(x, y)

    def b_minimum(self, node, x, y):
        return self.binary("min", x, y, node) if (_is_value(x) or _is_value(y)) \
            else builtins.min(x, y)

    def _unary(self, node, fn, x):
        x = self.to_value(x, node)
        if x.type.is_ptr:
            raise self.error(node, f"tl.{fn} does not take pointers")
        if fn in _FLOAT_ONLY_MATH and not x.type.elem.is_float:
            x = self.cast(x, tl.float32, node)
        return self.func.emit("unary", [x], x.type, fn=fn)

    def b_abs(self, node, x):
        return self._unary(node, "abs", x)

    def b_fma(self, node, x, y, z):
        return self.binary("add", self.binary("mul", x, y, node), z, node)

    def _reduce(self, node, kind, input, axis=None, keep_dims=False):
        x = self.to_value(input, node)
        if x.type.is_ptr:
            raise self.error(node, "cannot reduce pointers")
        if axis is not None:
            axis = self._const_int(axis, node, "axis")
            rank = len(x.type.shape)
            if axis < 0:
                axis += rank
            if not 0 <= axis < rank:
                raise self.error(node, f"axis {axis} out of range for rank {rank}")
        if kind == "sum" and (x.type.elem.is_bool or x.type.elem.bits < 32) \
                and not x.type.elem.is_float:
            x = self.cast(x, tl.int32, node)
        if axis is None:
            out_shape = ()
        else:
            out_shape = tuple(s for i, s in enumerate(x.type.shape) if i != axis)
        out = self.func.emit("reduce", [x], x.type.with_shape(out_shape), kind=kind, axis=axis)
        if keep_dims:
            if axis is None:
                for _ in x.type.shape:
                    out = self.expand_dims(out, 0, node)
            else:
                out = self.expand_dims(out, axis, node)
        return out

    def b_sum(self, node, input, axis=None, keep_dims=False):
        return self._reduce(node, "sum", input, axis, keep_dims)

    def b_max(self, node, input, axis=None, keep_dims=False, **kw):
        self._check_hints(node, kw, {"propagate_nan"})
        return self._reduce(node, "max", input, axis, keep_dims)

    def b_min(self, node, input, axis=None, keep_dims=False, **kw):
        self._check_hints(node, kw, {"propagate_nan"})
        return self._reduce(node, "min", input, axis, keep_dims)

    def b_cdiv(self, node, x, div):
        if not _is_value(x) and not _is_value(div):
            return tl._cdiv(x, div)
        return self.binary("floordiv", self.binary("sub", self.binary("add", x, div, node),
                                                   1, node), div, node)

    def b_static_range(self, node, *args):
        raise self.error(node, "tl.static_range can only be used as a for-loop iterator")

    def b_static_assert(self, node, cond, msg=""):
        if _is_value(cond):
            raise self.error(node, "tl.static_assert needs a compile-time condition")
        if not cond:
            raise self.error(node, f"static_assert failed: {msg}")

    def b_static_print(self, node, *args):
        print(*[a.type if _is_value(a) else a for a in args])

    def b_multiple_of(self, node, input, values):
        return input

    b_max_contiguous = b_multiple_of
    b_max_constancy = b_multiple_of


for _fn in _UNARY_MATH:
    setattr(FrontEnd, f"b_{_fn}", lambda self, node, x, _fn=_fn: self._unary(node, _fn, x))


def build_ir(fn, arg_types, constexprs, hook=None) -> ir.Function:
    fe = FrontEnd(fn, arg_types, constexprs)
    fe.hook = hook
    return fe.build()
