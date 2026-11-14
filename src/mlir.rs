use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use pyo3::prelude::*;

use crate::ir::{Attr, Elem, Function, Op};
use crate::CompileError;

const SIMD_WIDTH: usize = 32;

struct Frame {
    inits: Vec<usize>,
    args: Vec<Vec<String>>,
    results: Vec<Vec<String>>,
    types: Vec<String>,
}

fn mty(e: Elem) -> &'static str {
    match e {
        Elem::I1 => "i1",
        Elem::I32 => "i32",
        Elem::F32 => "f32",
    }
}

struct Gen<'a> {
    f: &'a Function,
    num_warps: usize,
    threads: usize,
    body: Vec<String>,
    workgroup: Vec<String>,
    names: HashMap<usize, Vec<String>>,
    base: HashMap<usize, String>,
    tmp: usize,
    n_reduce: usize,
    depth: usize,
    loops: Vec<Frame>,
    done: Vec<Frame>,
    def: HashMap<usize, usize>,
    cache: Vec<HashMap<String, String>>,
    n_dot: usize,
}

impl<'a> Gen<'a> {
    fn line(&mut self, s: String) {
        let pad = "  ".repeat(self.depth + 2);
        self.body.push(format!("{pad}{s}"));
    }

    fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }

    fn op(&mut self, rhs: String) -> String {
        let t = self.fresh();
        self.line(format!("{t} = {rhs}"));
        t
    }

    fn int(&mut self, v: i64) -> String {
        self.op(format!("arith.constant {v} : i32"))
    }

    fn idx(&mut self, v: i64) -> String {
        self.op(format!("arith.constant {v} : index"))
    }

    fn float(&mut self, v: f64) -> String {
        self.op(format!("arith.constant {v:.9e} : f32"))
    }

    fn to_index(&mut self, v: &str) -> String {
        self.op(format!("arith.index_cast {v} : i32 to index"))
    }

    fn regs(&self, v: usize) -> PyResult<usize> {
        let t = self.f.ty(v);
        if t.is_scalar() {
            return Ok(1);
        }
        Ok((t.numel() / self.threads).max(1))
    }

    fn replicated(&self, v: usize) -> Option<usize> {
        let t = self.f.ty(v);
        (!t.is_scalar() && t.numel() < self.threads).then(|| t.numel())
    }

    fn lookup(&self, key: &str) -> Option<String> {
        self.cache.iter().rev().find_map(|c| c.get(key).cloned())
    }

    fn remember(&mut self, key: String, value: String) -> String {
        self.cache.last_mut().unwrap().insert(key, value.clone());
        value
    }

    fn flat(&mut self, numel: usize, k: usize) -> String {
        let key = format!("flat:{numel}:{k}");
        if let Some(v) = self.lookup(&key) {
            return v;
        }
        let tid = self.tid();
        let v = if numel < self.threads {
            let c = self.int(numel as i64);
            self.op(format!("arith.remui {tid}, {c} : i32"))
        } else if k == 0 {
            tid
        } else {
            let c = self.int((k * self.threads) as i64);
            self.op(format!("arith.addi {tid}, {c} : i32"))
        };
        self.remember(key, v)
    }

    fn patch(&self, shape: &[usize]) -> Option<(usize, usize, usize)> {
        if shape.len() != 2 {
            return None;
        }
        let (rows, cols) = (shape[0], shape[1]);
        let numel = rows * cols;
        if numel < self.threads || numel % self.threads != 0 {
            return None;
        }
        let per = numel / self.threads;
        let pc = 4.min(cols).min(per);
        let pr = per / pc;
        if per % pc != 0 || cols % pc != 0 || rows % pr != 0 || (rows / pr) * (cols / pc) != self.threads {
            return None;
        }
        Some((pr, pc, cols / pc))
    }

    fn coords(&mut self, shape: &[usize], k: usize) -> Vec<String> {
        if let Some((pr, pc, across)) = self.patch(shape) {
            let (a, b) = (k / pc, k % pc);
            let mut out = Vec::new();
            for (i, (size, step, offset)) in [(pr, across, a), (pc, across, b)].into_iter().enumerate() {
                let key = format!("patch:{shape:?}:{k}:{i}");
                if let Some(v) = self.lookup(&key) {
                    out.push(v);
                    continue;
                }
                let tid = self.tid();
                let s = self.int(step as i64);
                let t = if i == 0 {
                    self.op(format!("arith.divui {tid}, {s} : i32"))
                } else {
                    self.op(format!("arith.remui {tid}, {s} : i32"))
                };
                let z = self.int(size as i64);
                let base = self.op(format!("arith.muli {t}, {z} : i32"));
                let o = self.int(offset as i64);
                let c = self.op(format!("arith.addi {base}, {o} : i32"));
                out.push(self.remember(key, c));
            }
            return out;
        }
        let numel: usize = shape.iter().product();
        let e = self.flat(numel, k);
        let mut out = Vec::new();
        for i in 0..shape.len() {
            let key = format!("coord:{shape:?}:{k}:{i}");
            if let Some(v) = self.lookup(&key) {
                out.push(v);
                continue;
            }
            let stride: usize = shape[i + 1..].iter().product();
            let mut c = e.clone();
            if stride > 1 {
                let s = self.int(stride as i64);
                c = self.op(format!("arith.divui {c}, {s} : i32"));
            }
            if i > 0 {
                let n = self.int(shape[i] as i64);
                c = self.op(format!("arith.remui {c}, {n} : i32"));
            }
            out.push(self.remember(key, c));
        }
        out
    }

    fn project(coords: &[Option<String>], to: &[usize]) -> Vec<Option<String>> {
        let off = coords.len() - to.len();
        to.iter().enumerate().map(|(j, &n)| if n == 1 { None } else { coords[off + j].clone() }).collect()
    }

    fn operand(&mut self, o: usize, shape: &[usize], k: usize) -> PyResult<String> {
        let t = self.f.ty(o).clone();
        if t.is_scalar() {
            return Ok(self.slot(o, 0));
        }
        if t.shape == shape {
            return Ok(self.slot(o, k));
        }
        let cs: Vec<Option<String>> = self.coords(shape, k).into_iter().map(Some).collect();
        let proj = Self::project(&cs, &t.shape);
        self.remat(o, &proj)
    }

    fn remat(&mut self, v: usize, coords: &[Option<String>]) -> PyResult<String> {
        let t = self.f.ty(v).clone();
        if t.is_scalar() {
            return Ok(self.slot(v, 0));
        }
        let key = format!("remat:{v}:{coords:?}");
        if let Some(r) = self.lookup(&key) {
            return Ok(r);
        }
        let op = &self.f.ops[*self.def.get(&v).ok_or_else(|| CompileError::new_err("cannot broadcast a kernel parameter"))?];
        let sub = |g: &mut Self, o: usize| -> PyResult<String> {
            let ot = g.f.ty(o).clone();
            if ot.is_scalar() {
                return Ok(g.slot(o, 0));
            }
            let off = coords.len() - ot.shape.len();
            g.remat(o, &coords[off..])
        };
        let r = match op.opcode {
            "arange" => {
                let Attr::Int(start) = op.attr("start") else { unreachable!() };
                let s = self.int(*start);
                match &coords[0] {
                    Some(c) => self.op(format!("arith.addi {c}, {s} : i32")),
                    None => s,
                }
            }
            "broadcast" => {
                let src = op.operands[0];
                let st = self.f.ty(src).clone();
                if st.is_scalar() {
                    self.slot(src, 0)
                } else {
                    let proj = Self::project(coords, &st.shape);
                    self.remat(src, &proj)?
                }
            }
            "expand_dims" => {
                let Attr::Int(axis) = op.attr("axis") else { unreachable!() };
                let mut c = coords.to_vec();
                c.remove(*axis as usize);
                self.remat(op.operands[0], &c)?
            }
            "binary" | "compare" | "addptr" => {
                let (a, b) = (op.operands[0], op.operands[1]);
                let (x, y) = (sub(self, a)?, sub(self, b)?);
                self.emit_binary(op, a, x, y)
            }
            other => return Err(CompileError::new_err(format!(
                "cannot broadcast the result of '{other}' to a larger shape; broadcast its inputs instead"))),
        };
        Ok(self.remember(key, r))
    }

    fn emit_binary(&mut self, op: &Op, a: usize, x: String, y: String) -> String {
        match op.opcode {
            "addptr" => self.op(format!("arith.addi {x}, {y} : i32")),
            "compare" => {
                let e = self.f.ty(a).elem;
                let (name, pred) = if e == Elem::F32 { ("arith.cmpf", "olt") } else { ("arith.cmpi", "slt") };
                self.op(format!("{name} {pred}, {x}, {y} : {}", mty(e)))
            }
            _ => {
                let e = self.f.ty(op.result.unwrap()).elem;
                let name = match (op.attr("kind"), e != Elem::F32) {
                    (Attr::Str("add"), true) => "arith.addi",
                    (Attr::Str("mul"), true) => "arith.muli",
                    (Attr::Str("and"), true) => "arith.andi",
                    (Attr::Str("add"), false) => "arith.addf",
                    (Attr::Str("mul"), false) => "arith.mulf",
                    _ => unreachable!(),
                };
                self.op(format!("{name} {x}, {y} : {}", mty(e)))
            }
        }
    }

    fn slot(&self, v: usize, r: usize) -> String {
        let n = &self.names[&v];
        if n.len() == 1 { n[0].clone() } else { n[r].clone() }
    }

    fn tid(&self) -> String {
        "%tid".to_string()
    }

    fn generate(mut self) -> PyResult<String> {
        let mut args = Vec::new();
        let mut arg_names = Vec::new();
        let zero = "%c0".to_string();
        // bc in metal every kernel argument arrives through a 
        // numbered buffer in the GPU memory. 
        // so every param needs it be a mlir memref

        // if its a pointer, names has the offsets
        // if not pointer, names has the actual value (SSA number)
        // that SSA holds the actual value loaded from the arg's memref
        for (i, (pname, v)) in self.f.params.iter().enumerate() {
            let t = self.f.ty(*v);
            let e = mty(t.elem);
            arg_names.push(format!("\"{pname}\""));
            if t.ptr {
                args.push(format!("%a{i}: memref<?x{e}>"));
                self.base.insert(*v, format!("%a{i}"));
                self.names.insert(*v, vec!["%zero".to_string()]);
            } else {
                args.push(format!("%a{i}: memref<1x{e}>"));
                let r = self.op(format!("memref.load %a{i}[{zero}] : memref<1x{e}>"));
                self.names.insert(*v, vec![r]);
            }
        }

        // a hashmap for which op number defines which tile IR ssa(result)
        for (i, op) in self.f.ops.iter().enumerate() {
            if let Some(r) = op.result {
                self.def.insert(r, i);
            }
        }

        for op in &self.f.ops {
            match op.opcode {
                "program_id" => {
                    let axis = match op.attr("axis") { Attr::Int(1) => "y", Attr::Int(2) => "z", _ => "x" };
                    //stores axis value to an mlir SSA
                    let b = self.op(format!("gpu.block_id {axis}"));
                    //converts the SSA to arith index
                    let r = self.op(format!("arith.index_cast {b} : index to i32"));
                    //maps that ssa to tile ir ssa
                    self.names.insert(op.result.unwrap(), vec![r]);
                }

                //creates new mlir ssa of type
                // of tile ir ssa and also creates a mapping in names
                "constant" => self.op_constant(op),

                // stores the positional values within 0...N 
                // that that thread is responsible for in names
                // (all of mlir is generated at thread level)
                "arange" => self.op_arange(op)?,

                //reshapes scalars and vectors
                "broadcast" => self.op_broadcast(op)?,


                "binary" => self.op_binary(op)?,
                "compare" => self.op_compare(op)?,
                "addptr" => self.op_addptr(op)?,

                "load" => self.op_load(op)?,
                "store" => self.op_store(op)?,
                "reduce" => self.op_reduce(op)?,
                "expand_dims" => self.op_expand_dims(op)?,
                "dot" => self.op_dot(op)?,
                "loop_begin" => self.op_loop_begin(op)?,
                "loop_arg" => {
                    let k = Self::index(op);
                    let r = op.result.unwrap();
                    let frame = self.loops.last().unwrap();
                    let (names, init) = (frame.args[k].clone(), frame.inits[k]);
                    if self.f.ty(r).ptr {
                        let b = self.base[&init].clone();
                        self.base.insert(r, b);
                    }
                    self.names.insert(r, names);
                }
                "loop_end" => {
                    let frame = self.loops.pop().unwrap();
                    if !frame.types.is_empty() {
                        let mut vals = Vec::new();
                        for ((y, a), init) in op.operands.iter().zip(&frame.args).zip(&frame.inits) {
                            if self.f.ty(*y).ptr && self.base[y] != self.base[init] {
                                return Err(CompileError::new_err("a pointer carried through a loop must stay in the same buffer"));
                            }
                            for k in 0..a.len() {
                                vals.push(self.slot(*y, k));
                            }
                        }
                        self.line(format!("scf.yield {} : {}", vals.join(", "), frame.types.join(", ")));
                    }
                    self.cache.pop();
                    self.depth -= 1;
                    self.line("}".to_string());
                    self.done.push(frame);
                }
                "loop_result" => {
                    let k = Self::index(op);
                    let (a, r) = (op.operands[0], op.result.unwrap());
                    if self.f.ty(a).ptr {
                        let b = self.base[&a].clone();
                        self.base.insert(r, b);
                    }
                    let names = self.done.last().unwrap().results[k].clone();
                    self.names.insert(r, names);
                }
                "barrier" => self.line("gpu.barrier".to_string()),
                other => return Err(CompileError::new_err(format!("MLIR backend: op '{other}' not supported"))),
            }
        }

        let cs: Vec<String> = self.f.constexprs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let wg = if self.workgroup.is_empty() { String::new() } else { format!(" workgroup({})", self.workgroup.join(", ")) };
        let mut out = vec![
            "module attributes {gpu.container_module} {".to_string(),
            "gpu.module @tilec {".to_string(),
            format!("  gpu.func @{}({}){wg} kernel attributes {{tilec.arg_names = [{}], tilec.info = \"{} threads/threadgroup  |  {}\"}} {{",
                    self.f.name, args.join(", "), arg_names.join(", "), self.threads, cs.join(", ")),
            "    %c0 = arith.constant 0 : index".to_string(),
            "    %zero = arith.constant 0 : i32".to_string(),
            "    %tidx = gpu.thread_id x".to_string(),
            "    %tid = arith.index_cast %tidx : index to i32".to_string(),
        ];
        out.extend(self.body);
        out.extend(["    gpu.return".to_string(), "  }".to_string(), "}".to_string(), "}".to_string()]);
        Ok(out.join("\n") + "\n")
    }

    fn index(op: &Op) -> usize {
        match op.attr("index") {
            Attr::Int(k) => *k as usize,
            _ => unreachable!(),
        }
    }

    fn slot_type(&self, v: usize) -> &'static str {
        let t = self.f.ty(v);
        if t.ptr { "i32" } else { mty(t.elem) }
    }

    fn op_loop_begin(&mut self, op: &Op) -> PyResult<()> {
        let (lo, hi, st, iv) = (op.operands[0], op.operands[1], op.operands[2], op.result.unwrap());
        let (lo, hi, st) = (self.slot(lo, 0), self.slot(hi, 0), self.slot(st, 0));
        let (lo, hi, one) = (self.to_index(&lo), self.to_index(&hi), self.to_index(&st));
        let i = self.fresh();
        let res = self.fresh();
        let mut frame = Frame { inits: op.operands[3..].to_vec(), args: vec![], results: vec![], types: vec![] };
        let mut inits = Vec::new();
        let mut count = 0;
        for &init in &op.operands[3..] {
            let n = self.regs(init)?;
            let ty = self.slot_type(init);
            let mut args = Vec::new();
            let mut results = Vec::new();
            for k in 0..n {
                let a = format!("{res}_a{count}");
                inits.push(format!("{a} = {}", self.slot(init, k)));
                frame.types.push(ty.to_string());
                args.push(a);
                results.push(count);
                count += 1;
            }
            frame.args.push(args);
            frame.results.push(results.iter().map(|c| c.to_string()).collect());
        }
        for r in frame.results.iter_mut() {
            for c in r.iter_mut() {
                *c = if count == 1 { res.clone() } else { format!("{res}#{c}") };
            }
        }
        if count == 0 {
            self.line(format!("scf.for {i} = {lo} to {hi} step {one} {{"));
        } else {
            let lhs = if count == 1 { res.clone() } else { format!("{res}:{count}") };
            self.line(format!("{lhs} = scf.for {i} = {lo} to {hi} step {one} iter_args({}) -> ({}) {{",
                              inits.join(", "), frame.types.join(", ")));
        }
        self.loops.push(frame);
        self.depth += 1;
        self.cache.push(HashMap::new());
        let r = self.op(format!("arith.index_cast {i} : index to i32"));
        self.names.insert(iv, vec![r]);
        Ok(())
    }

    fn op_constant(&mut self, op: &Op) {
        //the tile ir result ssa
        let r = op.result.unwrap();

        let e = self.f.ty(r).elem;
        let v = match (op.attr("value"), e) {
            (Attr::Int(i), Elem::I1) => self.op(format!("arith.constant {} : i1", *i != 0)),
            (Attr::Int(i), Elem::I32) => self.int(*i),
            (Attr::Int(i), Elem::F32) => self.float(*i as f64),
            (Attr::Float(x), Elem::F32) => self.float(*x),
            (Attr::Float(x), _) => self.int(*x as i64),
            (Attr::Str(s), _) => self.op(format!("arith.constant {s} : {}", mty(e))),
        };
        self.names.insert(r, vec![v]);
    }

    fn op_arange(&mut self, op: &Op) -> PyResult<()> {
        //the tile ir result ssa
        let r = op.result.unwrap();

        let Attr::Int(start) = op.attr("start") else { unreachable!() };
        let start = *start;
        let tid = self.tid();
        let mut slots = Vec::new();
        if let Some(n) = self.replicated(r) {
            let c = self.int(n as i64);
            let m = self.op(format!("arith.remui {tid}, {c} : i32"));
            let s = self.int(start);
            slots.push(self.op(format!("arith.addi {m}, {s} : i32")));
        } else {
            for k in 0..self.regs(r)? {
                let c = self.int(start + (k * self.threads) as i64);
                slots.push(self.op(format!("arith.addi {tid}, {c} : i32")));
            }
        }

        // elements of 0...N that the thread tid 
        // is responsible for gets stored in the mapping
        self.names.insert(r, slots);
        Ok(())
    }

    fn op_broadcast(&mut self, op: &Op) -> PyResult<()> {
        let (src, r) = (op.operands[0], op.result.unwrap());

        // copy base pointer ssa of it is of pointer type
        if self.f.ty(r).ptr {
            let b = self.base[&src].clone();
            self.base.insert(r, b);
        }

        // if its a single scalar(shape 1)
        if self.f.ty(src).is_scalar() {
            let s = self.slot(src, 0);
            let n = self.regs(r)?;
            self.names.insert(r, vec![s; n]);
            return Ok(());
        }

        // shaped scalar 
        let shape = self.f.ty(r).shape.clone();
        self.each(r, |g, k| g.operand(src, &shape, k))
    }

    fn op_expand_dims(&mut self, op: &Op) -> PyResult<()> {
        let (src, r) = (op.operands[0], op.result.unwrap());
        if self.f.ty(r).ptr {
            let b = self.base[&src].clone();
            self.base.insert(r, b);
        }
        let Attr::Int(axis) = op.attr("axis") else { unreachable!() };
        let axis = *axis as usize;
        let shape = self.f.ty(r).shape.clone();
        self.each(r, |g, k| {
            let mut cs: Vec<Option<String>> = g.coords(&shape, k).into_iter().map(Some).collect();
            cs.remove(axis);
            g.remat(src, &cs)
        })
    }

    fn each(&mut self, r: usize, mut f: impl FnMut(&mut Self, usize) -> PyResult<String>) -> PyResult<()> {
        let n = self.regs(r)?;
        let mut slots = Vec::with_capacity(n);
        for k in 0..n {
            slots.push(f(self, k)?);
        }
        self.names.insert(r, slots);
        Ok(())
    }

    fn op_binary(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        if self.f.ty(r).ptr {
            let base = self.base[&a].clone();
            self.base.insert(r, base);
        }
        let shape = self.f.ty(r).shape.clone();
        self.each(r, |g, k| {
            let (x, y) = (g.operand(a, &shape, k)?, g.operand(b, &shape, k)?);
            Ok(g.emit_binary(op, a, x, y))
        })
    }

    fn op_compare(&mut self, op: &Op) -> PyResult<()> {
        self.op_binary(op)
    }

    fn op_addptr(&mut self, op: &Op) -> PyResult<()> {
        self.op_binary(op)
    }

    fn op_load(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, mask, other, r) = (op.operands[0], op.operands[1], op.operands[2], op.result.unwrap());
        let base = self.base[&ptr].clone();
        let e = mty(self.f.ty(r).elem);
        let shape = self.f.ty(r).shape.clone();
        self.each(r, |g, k| {
            let (p, m, o) = (g.operand(ptr, &shape, k)?, g.operand(mask, &shape, k)?, g.operand(other, &shape, k)?);
            let safe = g.op(format!("arith.select {m}, {p}, %zero : i32"));
            let i = g.to_index(&safe);
            let x = g.op(format!("memref.load {base}[{i}] : memref<?x{e}>"));
            Ok(g.op(format!("arith.select {m}, {x}, {o} : {e}")))
        })
    }

    fn op_store(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, val) = (op.operands[0], op.operands[1]);
        let mask = op.operands.get(2).copied();
        let base = self.base[&ptr].clone();
        let e = mty(self.f.ty(val).elem);
        let shape = self.f.ty(val).shape.clone();
        let tid = self.tid();
        let owner = if self.f.ty(val).is_scalar() {
            Some(self.op(format!("arith.cmpi eq, {tid}, %zero : i32")))
        } else if let Some(n) = self.replicated(val) {
            let c = self.int(n as i64);
            Some(self.op(format!("arith.cmpi slt, {tid}, {c} : i32")))
        } else {
            None
        };
        for k in 0..self.regs(val)? {
            let (p, v) = (self.operand(ptr, &shape, k)?, self.operand(val, &shape, k)?);
            let guard = match (&owner, mask) {
                (Some(o), Some(m)) => {
                    let m = self.operand(m, &shape, k)?;
                    Some(self.op(format!("arith.andi {o}, {m} : i1")))
                }
                (Some(o), None) => Some(o.clone()),
                (None, Some(m)) => Some(self.operand(m, &shape, k)?),
                (None, None) => None,
            };
            let i = self.to_index(&p);
            let st = format!("memref.store {v}, {base}[{i}] : memref<?x{e}>");
            match guard {
                Some(gd) => {
                    self.line(format!("scf.if {gd} {{"));
                    self.line(format!("  {st}"));
                    self.line("}".to_string());
                }
                None => self.line(st),
            }
        }
        Ok(())
    }

    fn stage(&mut self, v: usize, buf: &str, wty: &str) -> PyResult<()> {
        let numel = self.f.ty(v).numel();
        let owner = match self.replicated(v) {
            Some(n) => {
                let c = self.int(n as i64);
                let tid = self.tid();
                Some(self.op(format!("arith.cmpi slt, {tid}, {c} : i32")))
            }
            None => None,
        };
        let shape = self.f.ty(v).shape.clone();
        for k in 0..self.regs(v)? {
            let e = if self.patch(&shape).is_some() {
                let cs = self.coords(&shape, k);
                let w = self.int(shape[1] as i64);
                let r = self.op(format!("arith.muli {}, {w} : i32", cs[0]));
                self.op(format!("arith.addi {r}, {} : i32", cs[1]))
            } else {
                self.flat(numel, k)
            };
            let i = self.to_index(&e);
            let x = self.slot(v, k);
            let st = format!("memref.store {x}, {buf}[{i}] : {wty}");
            match &owner {
                Some(o) => {
                    self.line(format!("scf.if {o} {{"));
                    self.line(format!("  {st}"));
                    self.line("}".to_string());
                }
                None => self.line(st),
            }
        }
        Ok(())
    }

    fn op_dot(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let (bm, bk) = (self.f.ty(a).shape[0], self.f.ty(a).shape[1]);
        let bn = self.f.ty(b).shape[1];
        let d = self.n_dot;
        self.n_dot += 1;
        let (abuf, bbuf) = (format!("%dot{d}a"), format!("%dot{d}b"));
        let aty = format!("memref<{}xf32, #gpu.address_space<workgroup>>", bm * bk);
        let bty = format!("memref<{}xf32, #gpu.address_space<workgroup>>", bk * bn);
        self.workgroup.push(format!("{abuf} : {aty}"));
        self.workgroup.push(format!("{bbuf} : {bty}"));

        self.line("gpu.barrier".to_string());
        self.stage(a, &abuf, &aty)?;
        self.stage(b, &bbuf, &bty)?;
        self.line("gpu.barrier".to_string());

        let shape = vec![bm, bn];
        let n = self.regs(r)?;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let bkc = self.int(bk as i64);
        for k in 0..n {
            let cs = self.coords(&shape, k);
            rows.push(self.op(format!("arith.muli {}, {bkc} : i32", cs[0])));
            cols.push(cs[1].clone());
        }
        let zero = self.float(0.0);
        let (lo, hi, one) = (self.idx(0), self.idx(bk as i64), self.idx(1));
        let bnc = self.int(bn as i64);
        let res = self.fresh();
        let kk = self.fresh();
        let accs: Vec<String> = (0..n).map(|j| format!("{res}_d{j}")).collect();
        let inits: Vec<String> = accs.iter().map(|x| format!("{x} = {zero}")).collect();
        let types = vec!["f32"; n].join(", ");
        let lhs = if n == 1 { res.clone() } else { format!("{res}:{n}") };
        self.line(format!("{lhs} = scf.for {kk} = {lo} to {hi} step {one} iter_args({}) -> ({types}) {{", inits.join(", ")));
        self.depth += 1;
        let kki = self.op(format!("arith.index_cast {kk} : index to i32"));
        let krow = self.op(format!("arith.muli {kki}, {bnc} : i32"));
        let mut next = Vec::new();
        for j in 0..n {
            let ai = self.op(format!("arith.addi {}, {kki} : i32", rows[j]));
            let ai = self.to_index(&ai);
            let av = self.op(format!("memref.load {abuf}[{ai}] : {aty}"));
            let bi = self.op(format!("arith.addi {krow}, {} : i32", cols[j]));
            let bi = self.to_index(&bi);
            let bv = self.op(format!("memref.load {bbuf}[{bi}] : {bty}"));
            let p = self.op(format!("arith.mulf {av}, {bv} : f32"));
            next.push(self.op(format!("arith.addf {}, {p} : f32", accs[j])));
        }
        self.line(format!("scf.yield {} : {types}", next.join(", ")));
        self.depth -= 1;
        self.line("}".to_string());
        let names = (0..n).map(|j| if n == 1 { res.clone() } else { format!("{res}#{j}") }).collect();
        self.names.insert(r, names);
        Ok(())
    }

    fn op_reduce(&mut self, op: &Op) -> PyResult<()> {
        let (x, r) = (op.operands[0], op.result.unwrap());
        let el = self.f.ty(x).elem;
        let e = mty(el);
        let (add, zero) = if el == Elem::F32 { ("arith.addf", self.float(0.0)) } else { ("arith.addi", self.int(0)) };
        let k = self.n_reduce;
        self.n_reduce += 1;
        let nw = self.num_warps;
        let wty = format!("memref<{nw}x{e}, #gpu.address_space<workgroup>>");
        let red = format!("%red{k}");
        self.workgroup.push(format!("{red} : {wty}"));
        let tid = self.tid();
        let owner = match self.replicated(x) {
            Some(n) => {
                let c = self.int(n as i64);
                Some(self.op(format!("arith.cmpi slt, {tid}, {c} : i32")))
            }
            None => None,
        };
        let mut acc = zero.clone();
        for s in 0..self.regs(x)? {
            let mut v = self.slot(x, s);
            if let Some(o) = &owner {
                v = self.op(format!("arith.select {o}, {v}, {zero} : {e}"));
            }
            acc = self.op(format!("{add} {acc}, {v} : {e}"));
        }
        let acc = self.op(format!("gpu.subgroup_reduce add {acc} : ({e}) -> {e}"));
        let w = self.int(SIMD_WIDTH as i64);
        let lane = self.op(format!("arith.remui {tid}, {w} : i32"));
        let sg = self.op(format!("arith.divui {tid}, {w} : i32"));
        let first = self.op(format!("arith.cmpi eq, {lane}, %zero : i32"));
        let sgi = self.to_index(&sg);
        self.line(format!("scf.if {first} {{"));
        self.line(format!("  memref.store {acc}, {red}[{sgi}] : {wty}"));
        self.line("}".to_string());
        self.line("gpu.barrier".to_string());
        let mut total = self.op(format!("memref.load {red}[%c0] : {wty}"));
        for w in 1..nw {
            let i = self.idx(w as i64);
            let p = self.op(format!("memref.load {red}[{i}] : {wty}"));
            total = self.op(format!("{add} {total}, {p} : {e}"));
        }
        if self.depth > 0 {
            self.line("gpu.barrier".to_string());
        }
        self.names.insert(r, vec![total]);
        Ok(())
    }
}

pub fn generate(f: &Function, num_warps: usize) -> PyResult<String> {
    Gen {
        f,
        num_warps,
        threads: num_warps * SIMD_WIDTH,
        body: vec![],
        workgroup: vec![],
        names: HashMap::new(),
        base: HashMap::new(),
        tmp: 0,
        n_reduce: 0,
        depth: 0,
        loops: vec![],
        done: vec![],
        def: HashMap::new(),
        cache: vec![HashMap::new()],
        n_dot: 0,
    }
    .generate()
}

pub fn lower(tool: &str, mlir: &str) -> PyResult<(String, String)> {
    let mut child = Command::new(tool)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CompileError::new_err(format!(
            "cannot run {tool}: {e}\nbuild it with `ninja -C mlir/build`, or set TILEC_OPT= (empty) to use the built-in Metal backend")))?;
    child.stdin.take().unwrap().write_all(mlir.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(CompileError::new_err(format!("{tool} failed:\n{}\ninput:\n{mlir}",
                                                 String::from_utf8_lossy(&out.stderr))));
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let Some((opt, msl)) = text.split_once("//--MSL--\n") else {
        return Err(CompileError::new_err(format!("{tool} printed no Metal")));
    };
    Ok((opt.to_string(), msl.to_string()))
}
