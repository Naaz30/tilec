use std::collections::HashMap;

use pyo3::prelude::*;

use crate::ir::{Attr, Function, Op};
use crate::CompileError;

const SIMD_WIDTH: usize = 32;

fn slot(v: usize, scalar: bool, r: &str) -> String {
    if scalar { format!("v{v}") } else { format!("v{v}[{r}]") }
}

struct Layout {
    n: usize,
    threads: usize,
}

impl Layout {
    fn regs(&self) -> usize {
        (self.n / self.threads).max(1)
    }
    fn replicated(&self) -> bool {
        self.n < self.threads
    }

    fn element(&self, r: &str) -> String {
        if self.replicated() {
            format!("(tid % {}u)", self.n)
        } else {
            format!("({r} * {}u + tid)", self.threads)
        }
    }

    fn owner_guard(&self) -> Option<String> {
        self.replicated().then(|| format!("(tid < {}u)", self.n))
    }
}

struct Gen<'a> {
    f: &'a Function,
    num_warps: usize,
    threads: usize,
    body: Vec<String>,
    shared: Vec<String>,              
    ptr_base: HashMap<usize, String>, 
    n_reduce: usize,
    depth: usize, 
    loops: Vec<Vec<usize>>,
}

impl<'a> Gen<'a> {
    fn layout(&self, v: usize) -> PyResult<Layout> {
        let shape = &self.f.ty(v).shape;
        if shape.len() != 1 {
            return Err(CompileError::new_err("2-D tiles need the MLIR backend (unset TILEC_OPT or point it at tilec-opt)"));
        }
        Ok(Layout { n: shape[0], threads: self.threads })
    }

    fn rf(&self, v: usize, r: &str) -> String {
        slot(v, self.f.ty(v).is_scalar(), r)
    }

    fn scalar(&self, v: usize) -> bool {
        self.f.ty(v).is_scalar()
    }

    fn ctype(&self, v: usize) -> &'static str {
        let t = self.f.ty(v);
        if t.ptr { "int" } else { t.elem.msl() }
    }

    fn emit(&mut self, line: String) {
        let indent = "    ".repeat(self.depth + 1);
        self.body.push(format!("{indent}{line}"));
    }

    fn define(&mut self, v: usize, expr: impl Fn(&str) -> String) -> PyResult<()> {
        let ct = self.ctype(v);
        if self.f.ty(v).is_scalar() {
            self.emit(format!("{ct} v{v} = {};", expr("0")));
            return Ok(());
        }
        let regs = self.layout(v)?.regs();
        self.emit(format!("{ct} v{v}[{regs}];"));
        self.emit(format!("for (uint r = 0; r < {regs}u; ++r) v{v}[r] = {};", expr("r")));
        Ok(())
    }

    fn copy(&mut self, dst: usize, src: usize) -> PyResult<()> {
        if self.f.ty(src).ptr {
            let b = self.base(src);
            self.ptr_base.insert(dst, b);
        }
        let scalar = self.scalar(src);
        self.define(dst, move |r| slot(src, scalar, r))
    }

    fn copy_named(&mut self, dst: &str, like: usize, src: usize, declare: bool) -> PyResult<()> {
        let ct = self.ctype(like);
        if self.scalar(like) {
            let decl = if declare { format!("{ct} ") } else { String::new() };
            self.emit(format!("{decl}{dst} = v{src};"));
            return Ok(());
        }
        let regs = self.layout(like)?.regs();
        if declare {
            self.emit(format!("{ct} {dst}[{regs}];"));
        }
        self.emit(format!("for (uint r = 0; r < {regs}u; ++r) {dst}[r] = v{src}[r];"));
        Ok(())
    }

    fn assign(&mut self, dst: &str, src: &str, like: usize) -> PyResult<()> {
        if self.scalar(like) {
            self.emit(format!("{dst} = {src};"));
            return Ok(());
        }
        let regs = self.layout(like)?.regs();
        self.emit(format!("for (uint r = 0; r < {regs}u; ++r) {dst}[r] = {src}[r];"));
        Ok(())
    }

    fn base(&self, ptr: usize) -> String {
        self.ptr_base[&ptr].clone()
    }

    fn generate(mut self) -> PyResult<String> {
        let mut params = Vec::new();
        for (i, (pname, v)) in self.f.params.iter().enumerate() {
            let (v, t) = (*v, self.f.ty(*v));
            if t.ptr {
                params.push(format!("device {}* {pname} [[buffer({i})]]", t.elem.msl()));
                self.ptr_base.insert(v, pname.clone());
                self.emit(format!("int v{v} = 0;  // {pname}"));
            } else {
                let ct = t.elem.msl();
                params.push(format!("device const {ct}* {pname}_arg [[buffer({i})]]"));
                self.emit(format!("const {ct} v{v} = {pname}_arg[0];  // {pname}"));
            }
        }
        params.extend([
            "uint3 tgpig [[threadgroup_position_in_grid]]".to_string(),
            "uint3 tid3 [[thread_position_in_threadgroup]]".to_string(),
            "uint lane [[thread_index_in_simdgroup]]".to_string(),
            "uint sg [[simdgroup_index_in_threadgroup]]".to_string(),
        ]);
        self.body.insert(0, "    const uint tid = tid3.x;".to_string());

        for (i, op) in self.f.ops.iter().enumerate() {
            match op.opcode {
                "program_id" => self.op_program_id(op),
                "constant" => self.op_constant(op),
                "arange" => self.op_arange(op)?,
                "broadcast" => self.op_broadcast(op)?,
                "binary" => self.op_binary(op)?,
                "compare" => self.op_compare(op)?,
                "addptr" => self.op_addptr(op)?,
                "load" => self.op_load(op)?,
                "store" => self.op_store(op)?,
                "reduce" => self.op_reduce(op)?,
                "loop_begin" => {
                    let (start, end, step, iv) = (op.operands[0], op.operands[1], op.operands[2], op.result.unwrap());
                    let args: Vec<usize> = self.f.ops[i + 1..].iter()
                        .take_while(|o| o.opcode == "loop_arg")
                        .map(|o| o.result.unwrap())
                        .collect();
                    for (a, init) in args.iter().zip(&op.operands[3..]) {
                        self.copy(*a, *init)?;
                    }
                    self.loops.push(args);
                    self.emit(format!("for (int v{iv} = v{start}; v{iv} < v{end}; v{iv} += v{step}) {{"));
                    self.depth += 1;
                }
                "loop_arg" => {}
                "loop_end" => {
                    let args = self.loops.pop().unwrap();
                    let temp = op.operands.iter().any(|y| args.contains(y));
                    for (a, y) in args.iter().zip(&op.operands) {
                        if temp {
                            self.copy_named(&format!("t{a}"), *a, *y, true)?;
                        } else {
                            self.copy_named(&format!("v{a}"), *a, *y, false)?;
                        }
                    }
                    if temp {
                        for a in &args {
                            let (dst, src) = (format!("v{a}"), format!("t{a}"));
                            self.assign(&dst, &src, *a)?;
                        }
                    }
                    self.depth -= 1;
                    self.emit("}".to_string());
                }
                "loop_result" => self.copy(op.result.unwrap(), op.operands[0])?,
                "barrier" => self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()),
                "expand_dims" | "dot" => return Err(CompileError::new_err(format!(
                    "op '{}' needs the MLIR backend (unset TILEC_OPT or point it at tilec-opt)", op.opcode))),
                other => return Err(CompileError::new_err(format!("Metal backend: op '{other}' not supported"))),
            }
        }

        let cs: Vec<String> = self.f.constexprs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let mut out = vec![
            "#include <metal_stdlib>".to_string(),
            "using namespace metal;".to_string(),
            String::new(),
            format!("// generated by tilec  |  {} threads/threadgroup  |  {}", self.threads, cs.join(", ")),
            format!("kernel void {}(", self.f.name),
            format!("    {})", params.join(",\n    ")),
            "{".to_string(),
        ];
        out.extend(self.shared.iter().map(|s| format!("    {s}")));
        out.extend(self.body);
        out.push("}".to_string());
        Ok(out.join("\n") + "\n")
    }

    fn op_program_id(&mut self, op: &Op) {
        let r = op.result.unwrap();
        let axis = match op.attr("axis") { Attr::Int(1) => "y", Attr::Int(2) => "z", _ => "x" };
        self.emit(format!("const int v{r} = int(tgpig.{axis});"));
    }

    fn op_constant(&mut self, op: &Op) {
        let r = op.result.unwrap();
        let t = self.f.ty(r);
        let lit = match op.attr("value") {
            Attr::Int(i) => format!("{}({i})", t.elem.msl()),
            Attr::Float(x) => format!("{}({x:?}f)", t.elem.msl()),
            Attr::Str(s) => s.to_string(),
        };
        self.emit(format!("const {} v{r} = {lit};", t.elem.msl()));
    }

    fn op_arange(&mut self, op: &Op) -> PyResult<()> {
        let r = op.result.unwrap();
        let element = self.layout(r)?.element("r");
        let Attr::Int(start) = op.attr("start") else { unreachable!() };
        let start = *start;
        self.define(r, |_| format!("{start} + int{element}"))
    }

    fn op_broadcast(&mut self, op: &Op) -> PyResult<()> {
        let (src, r) = (op.operands[0], op.result.unwrap());
        if !self.f.ty(src).is_scalar() {
            return Err(CompileError::new_err("the Metal backend only broadcasts scalars"));
        }
        if self.f.ty(r).ptr {
            let b = self.base(src);
            self.ptr_base.insert(r, b);
        }
        let s = self.rf(src, "r");
        self.define(r, |_| s.clone())
    }

    fn op_binary(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let sym = match op.attr("kind") { Attr::Str("add") => "+", Attr::Str("mul") => "*", Attr::Str("and") => "&", _ => unreachable!() };
        let ct = self.ctype(r);
        let (sa, sb) = (self.scalar(a), self.scalar(b));
        self.define(r, |r| format!("{ct}({} {sym} {})", slot(a, sa, r), slot(b, sb, r)))
    }

    fn op_compare(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let (sa, sb) = (self.scalar(a), self.scalar(b));
        self.define(r, |r| format!("({} < {})", slot(a, sa, r), slot(b, sb, r)))
    }

    fn op_addptr(&mut self, op: &Op) -> PyResult<()> {
        let (p, off, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let b = self.base(p);
        self.ptr_base.insert(r, b);
        let (sp, so) = (self.scalar(p), self.scalar(off));
        self.define(r, |r| format!("{} + {}", slot(p, sp, r), slot(off, so, r)))
    }

    fn op_load(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, mask, other, r) = (op.operands[0], op.operands[1], op.operands[2], op.result.unwrap());
        let base = self.base(ptr);
        let (sm, sp, so) = (self.scalar(mask), self.scalar(ptr), self.scalar(other));
        self.define(r, |r| format!("({} ? {base}[{}] : {})", slot(mask, sm, r), slot(ptr, sp, r), slot(other, so, r)))
    }

    fn op_store(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, val) = (op.operands[0], op.operands[1]);
        let base = self.base(ptr);
        let mask = op.operands.get(2).copied();
        if self.f.ty(val).is_scalar() {                
            let m = mask.map(|m| format!(" && v{m}")).unwrap_or_default();
            self.emit(format!("if (tid == 0u{m}) {base}[v{ptr}] = v{val};"));
            return Ok(());
        }
        let lay = self.layout(val)?;
        let mut stmt = format!("{base}[v{ptr}[r]] = v{val}[r];");
        let mut conds: Vec<String> = lay.owner_guard().into_iter().collect();
        if let Some(m) = mask {
            conds.push(slot(m, self.scalar(m), "r"));
        }
        if !conds.is_empty() {
            stmt = format!("if ({}) {stmt}", conds.join(" && "));
        }
        self.emit(format!("for (uint r = 0; r < {}u; ++r) {stmt}", lay.regs()));
        Ok(())
    }

    fn op_reduce(&mut self, op: &Op) -> PyResult<()> {
        let (x, r) = (op.operands[0], op.result.unwrap());
        let ct = self.f.ty(x).elem.msl();
        let lay = self.layout(x)?;
        let k = self.n_reduce;
        self.n_reduce += 1;
        let (acc, scratch, nw) = (format!("acc{k}"), format!("red{k}"), self.num_warps);
        self.shared.push(format!("threadgroup {ct} {scratch}[{nw}];"));
        let mut body = format!("{acc} = {acc} + v{x}[r];");
        if let Some(g) = lay.owner_guard() {          
            body = format!("if ({g}) {body}");
        }
        self.emit(format!("// sum over {} elements", self.f.ty(x).numel()));
        self.emit(format!("{ct} {acc} = {ct}(0);"));
        self.emit(format!("for (uint r = 0; r < {}u; ++r) {body}", lay.regs()));     
        self.emit(format!("{acc} = simd_sum({acc});"));                              
        self.emit(format!("if (lane == 0u) {scratch}[sg] = {acc};"));               
        self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);".to_string());
        self.emit(format!("{acc} = {scratch}[0];"));                                
        self.emit(format!("for (uint w = 1; w < {nw}u; ++w) {acc} = {acc} + {scratch}[w];"));
        if self.depth > 0 {
            self.emit("threadgroup_barrier(mem_flags::mem_threadgroup);".to_string());
        }
        self.emit(format!("const {ct} v{r} = {acc};"));
        Ok(())
    }
}

pub fn generate(f: &Function, num_warps: usize) -> PyResult<String> {
    Gen {
        f,
        num_warps,
        threads: num_warps * SIMD_WIDTH,
        body: vec![],
        shared: vec![],
        ptr_base: HashMap::new(),
        n_reduce: 0,
        depth: 0,
        loops: vec![],
    }
    .generate()
}