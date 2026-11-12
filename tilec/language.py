class constexpr:
    def __init__(self, value):
        self.value = value


class Builtin:
    def __init__(self, name):
        self.tilec_builtin = name         

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"tl.{self.tilec_builtin} can only be called inside a @jit kernel")

for _name in ["program_id", "arange", "load", "store", "sum", "static_range", "sum", "mul", "barrier", "zeros", "dot"]:
    globals()[_name] = Builtin(_name)

def cdiv(a, b):
    """Ceiling division, for host code: cdiv(5000, 4096) == 2."""
    return (a + b - 1) // b