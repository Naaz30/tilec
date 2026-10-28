"""A small SSA tile IR.

Every value is either a scalar or an N-D tile (a "tensor" in Triton terms) of
scalars or pointers.  Ops are kept in one flat list because Tier 1 has no
runtime control flow (constexpr ``if`` and ``static_range`` are resolved by
the front end).  Runtime loops will add region-carrying ops later.

Op catalogue (operands -> result):

    program_id   ()                         attrs axis           -> i32
    num_programs ()                         attrs axis           -> i32
    constant     ()                         attrs value          -> scalar
    arange       ()                         attrs start, end     -> i32[N]
    broadcast    (x)                        result shape         -> x.elem[shape]
    expand_dims  (x)                        attrs axis           -> rank + 1
    cast         (x)                        result elem          -> new elem
    binary       (a, b)                     attrs kind           -> same elem
    compare      (a, b)                     attrs kind           -> i1
    select       (cond, a, b)                                    -> a's type
    unary        (x)                        attrs fn             -> same type
    addptr       (ptr, offset)                                   -> ptr
    load         (ptr, mask?, other?)       attrs has_mask, has_other
    store        (ptr, value, mask?)        attrs has_mask       -> none
    reduce       (x)                        attrs kind, axis     -> smaller rank
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import language as tl


@dataclass(frozen=True)
class PtrType:
    elem: tl.dtype

    def __str__(self):
        return f"ptr<{self.elem.name}>"


@dataclass(frozen=True)
class Type:
    elem: object          
    shape: tuple = ()

    @property
    def is_ptr(self):
        return isinstance(self.elem, PtrType)

    @property
    def is_scalar(self):
        return self.shape == ()

    @property
    def dtype(self):
        if self.is_ptr:
            raise TypeError("pointer type has no scalar dtype")
        return self.elem

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def with_shape(self, shape):
        return Type(self.elem, tuple(shape))

    def with_elem(self, elem):
        return Type(elem, self.shape)

    def __str__(self):
        if self.is_scalar:
            return str(self.elem)
        return "tensor<" + "x".join(str(s) for s in self.shape) + f"x{self.elem}>"


class Value:
    def __init__(self, id_, type_, hint=None):
        self.id = id_
        self.type = type_
        self.hint = hint

    def __str__(self):
        return f"%{self.id}"

    def __repr__(self):
        return f"Value(%{self.id}: {self.type})"


@dataclass
class Op:
    opcode: str
    operands: list
    attrs: dict = field(default_factory=dict)
    result: Value | None = None
    loc: tuple | None = None  

    def __str__(self):
        args = ", ".join(str(o) for o in self.operands)
        attrs = ""
        if self.attrs:
            attrs = " {" + ", ".join(f"{k}={v!r}" for k, v in self.attrs.items()) + "}"
        lhs = f"{self.result} = " if self.result is not None else ""
        ty = f" : {self.result.type}" if self.result is not None else ""
        return f"{lhs}{self.opcode}({args}){attrs}{ty}"


class Function:
    def __init__(self, name):
        self.name = name
        self.params: list[tuple[str, Value]] = []
        self.constexprs: dict[str, object] = {}
        self.ops: list[Op] = []
        self._next = 0
        self.cur_loc = None       

    def _new_value(self, type_, hint=None):
        v = Value(self._next, type_, hint)
        self._next += 1
        return v

    def add_param(self, name, type_):
        v = self._new_value(type_, name)
        self.params.append((name, v))
        return v

    def emit(self, opcode, operands, type_=None, **attrs):
        result = self._new_value(type_) if type_ is not None else None
        self.ops.append(Op(opcode, list(operands), attrs, result, self.cur_loc))
        return result

    def format(self, with_locs=False):
        params = ", ".join(f"{v}: {v.type}  /* {n} */" for n, v in self.params)
        lines = [f"func @{self.name}({params})"]
        if self.constexprs:
            cs = ", ".join(f"{k}={v!r}" for k, v in self.constexprs.items())
            lines.append(f"  // constexpr: {cs}")
        if not with_locs:
            lines += ["  " + str(op) for op in self.ops]
            return "\n".join(lines)
        width = max((len(str(op)) for op in self.ops), default=0)
        for op in self.ops:
            loc = f"  // line {op.loc[0]}" if op.loc else ""
            lines.append(f"  {str(op):<{width}}{loc}")
        return "\n".join(lines)

    def __str__(self):
        return self.format()
