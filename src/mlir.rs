use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use pyo3::prelude::*;

use crate::ir::{Attr, Elem, Function, Op};
use crate::CompileError;

const SIMD_WIDTH: usize = 32;

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
        if t.shape.len() != 1 {
            return Err(CompileError::new_err("the MLIR backend only supports 1-D tiles"));
        }
        Ok((t.shape[0] / self.threads).max(1))
    }

    fn replicated(&self, v: usize) -> Option<usize> {
        let t = self.f.ty(v);
        (!t.is_scalar() && t.shape[0] < self.threads).then(|| t.shape[0])
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

        for op in &self.f.ops {
            match op.opcode {
                "program_id" => {
                    let b = self.op("gpu.block_id x".to_string());
                    let r = self.op(format!("arith.index_cast {b} : index to i32"));
                    self.names.insert(op.result.unwrap(), vec![r]);
                }
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
                    let (lo, hi, iv) = (op.operands[0], op.operands[1], op.result.unwrap());
                    let (lo, hi) = (self.slot(lo, 0), self.slot(hi, 0));
                    let (lo, hi, one) = (self.to_index(&lo), self.to_index(&hi), self.idx(1));
                    let i = self.fresh();
                    self.line(format!("scf.for {i} = {lo} to {hi} step {one} {{"));
                    self.depth += 1;
                    let r = self.op(format!("arith.index_cast {i} : index to i32"));
                    self.names.insert(iv, vec![r]);
                }
                "loop_end" => {
                    self.depth -= 1;
                    self.line("}".to_string());
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

    fn op_constant(&mut self, op: &Op) {
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
        self.names.insert(r, slots);
        Ok(())
    }

    fn op_broadcast(&mut self, op: &Op) -> PyResult<()> {
        let (src, r) = (op.operands[0], op.result.unwrap());
        if !self.f.ty(src).is_scalar() {
            return Err(CompileError::new_err("the MLIR backend only broadcasts scalars"));
        }
        if self.f.ty(r).ptr {
            let b = self.base[&src].clone();
            self.base.insert(r, b);
        }
        let s = self.slot(src, 0);
        let n = self.regs(r)?;
        self.names.insert(r, vec![s; n]);
        Ok(())
    }

    fn each(&mut self, r: usize, mut f: impl FnMut(&mut Self, usize) -> String) -> PyResult<()> {
        let n = self.regs(r)?;
        let slots = (0..n).map(|k| f(self, k)).collect();
        self.names.insert(r, slots);
        Ok(())
    }

    fn op_binary(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let e = self.f.ty(r).elem;
        let name = match (op.attr("kind"), e.is_int() || e == Elem::I1) {
            (Attr::Str("add"), true) => "arith.addi",
            (Attr::Str("mul"), true) => "arith.muli",
            (Attr::Str("add"), false) => "arith.addf",
            (Attr::Str("mul"), false) => "arith.mulf",
            _ => unreachable!(),
        };
        let t = mty(e);
        self.each(r, |g, k| {
            let (x, y) = (g.slot(a, k), g.slot(b, k));
            g.op(format!("{name} {x}, {y} : {t}"))
        })
    }

    fn op_compare(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let e = self.f.ty(a).elem;
        let (name, pred) = if e == Elem::F32 { ("arith.cmpf", "olt") } else { ("arith.cmpi", "slt") };
        let t = mty(e);
        self.each(r, |g, k| {
            let (x, y) = (g.slot(a, k), g.slot(b, k));
            g.op(format!("{name} {pred}, {x}, {y} : {t}"))
        })
    }

    fn op_addptr(&mut self, op: &Op) -> PyResult<()> {
        let (p, off, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let b = self.base[&p].clone();
        self.base.insert(r, b);
        self.each(r, |g, k| {
            let (x, y) = (g.slot(p, k), g.slot(off, k));
            g.op(format!("arith.addi {x}, {y} : i32"))
        })
    }

    fn op_load(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, mask, other, r) = (op.operands[0], op.operands[1], op.operands[2], op.result.unwrap());
        let base = self.base[&ptr].clone();
        let e = mty(self.f.ty(r).elem);
        self.each(r, |g, k| {
            let (p, m, o) = (g.slot(ptr, k), g.slot(mask, k), g.slot(other, k));
            let safe = g.op(format!("arith.select {m}, {p}, %zero : i32"));
            let i = g.to_index(&safe);
            let x = g.op(format!("memref.load {base}[{i}] : memref<?x{e}>"));
            g.op(format!("arith.select {m}, {x}, {o} : {e}"))
        })
    }

    fn op_store(&mut self, op: &Op) -> PyResult<()> {
        let (ptr, val) = (op.operands[0], op.operands[1]);
        let base = self.base[&ptr].clone();
        let e = mty(self.f.ty(val).elem);
        let tid = self.tid();
        let guard = if self.f.ty(val).is_scalar() {
            Some(self.op(format!("arith.cmpi eq, {tid}, %zero : i32")))
        } else if let Some(n) = self.replicated(val) {
            let c = self.int(n as i64);
            Some(self.op(format!("arith.cmpi slt, {tid}, {c} : i32")))
        } else {
            None
        };
        for k in 0..self.regs(val)? {
            let (p, v) = (self.slot(ptr, k), self.slot(val, k));
            let i = self.to_index(&p);
            let st = format!("memref.store {v}, {base}[{i}] : memref<?x{e}>");
            match &guard {
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
    }
    .generate()
}

pub fn lower(tool: &str, mlir: &str) -> PyResult<(String, String)> {
    let mut child = Command::new(tool)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CompileError::new_err(format!("cannot run {tool}: {e}")))?;
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
