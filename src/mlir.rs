use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use pyo3::prelude::*;

use crate::ir::{Attr, Elem, Function, Op};
use crate::CompileError;

const SIMD_WIDTH: usize = 32;
const WMITER: usize = 2;
const AMAT: &str = "!gpu.mma_matrix<8x8xf32, \"AOp\">";
const BMAT: &str = "!gpu.mma_matrix<8x8xf32, \"BOp\">";
const CMAT: &str = "!gpu.mma_matrix<8x8xf32, \"COp\">";

struct Frame {
    inits: Vec<usize>,
    args: Vec<Vec<String>>,
    results: Vec<Vec<String>>,
    types: Vec<String>,
}

type Sc = (String, Option<i64>);

#[derive(Clone)]
struct Aff {
    base: Sc,
    coef: Vec<Sc>,
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
    threads: usize,
    body: Vec<String>,
    workgroup: Vec<String>,
    names: HashMap<usize, Vec<String>>,
    base: HashMap<usize, String>,
    tmp: usize,
    depth: usize,
    loops: Vec<Frame>,
    done: Vec<Frame>,
    def: HashMap<usize, usize>,
    cache: Vec<HashMap<String, String>>,
    n_dot: usize,
    acc_loops: HashMap<usize, Vec<usize>>,
    mma_dot: HashMap<usize, usize>,
    mma_add: HashMap<usize, usize>,
    mma_out: HashMap<usize, usize>,
    dot_buf: HashMap<usize, String>,
    vec_src: HashMap<String, (String, usize)>,
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

    fn bool(&mut self, v: i64) -> String {
        self.op(format!("arith.constant {} : i1", v != 0))
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

    // splits a 2-D tile of a threadgroup into 2x2 quadrants, one for each SIMD-group
    // each quadrant is split into equal 2-D patches, one per thread (nx4).
    // each patch is split into pieces of ph rows, stride rows apart.
    // (corner row, corner col, [pr, pc], [ph, stride]), or None when the tile
    // can't be split this way.
    fn patch(&mut self, shape: &[usize]) -> Option<(String, String, [usize; 2], [usize; 2])> {
        if shape.len() != 2
            || self.threads != 4 * SIMD_WIDTH
            || shape[0] % 2 != 0
            || shape[1] % 2 != 0
        {
            return None;
        }
        let (qr, qc) = (shape[0] / 2, shape[1] / 2);
        let numel = qr * qc;
        if numel < SIMD_WIDTH || numel % SIMD_WIDTH != 0 {
            return None;
        }
        let per = numel / SIMD_WIDTH;
        let pc = 4.min(qc).min(per);
        let pr = per / pc;
        if per % pc != 0 || qc % pc != 0 || qr % pr != 0 || (qr / pr) * (qc / pc) != SIMD_WIDTH {
            return None;
        }
        let across = qc / pc;
        let sub = if pr % WMITER == 0 { WMITER } else { 1 };
        let (ph, stride) = (pr / sub, qr / sub);

        let mut corner = Vec::new();
        for (i, (q, size)) in [(qr, ph), (qc, pc)].into_iter().enumerate() {
            let key = format!("corner:{shape:?}:{i}");
            if let Some(v) = self.lookup(&key) {
                corner.push(v);
                continue;
            }
            let tid = self.tid();
            let w = self.int(SIMD_WIDTH as i64);
            let sg = self.op(format!("arith.divui {tid}, {w} : i32"));
            let lane = self.op(format!("arith.remui {tid}, {w} : i32"));
            let two = self.int(2);
            let s = self.int(across as i64);
            let (qpos, t) = if i == 0 {
                (
                    self.op(format!("arith.divui {sg}, {two} : i32")),
                    self.op(format!("arith.divui {lane}, {s} : i32")),
                )
            } else {
                (
                    self.op(format!("arith.remui {sg}, {two} : i32")),
                    self.op(format!("arith.remui {lane}, {s} : i32")),
                )
            };
            let qs = self.int(q as i64);
            let qbase = self.op(format!("arith.muli {qpos}, {qs} : i32"));
            let z = self.int(size as i64);
            let pbase = self.op(format!("arith.muli {t}, {z} : i32"));
            let c = self.op(format!("arith.addi {qbase}, {pbase} : i32"));
            corner.push(self.remember(key, c));
        }
        Some((corner[0].clone(), corner[1].clone(), [pr, pc], [ph, stride]))
    }

    // returns what the kth value of this thread point to
    // in the tile. Which element the kth slot of the thead
    // holds depends on the tile's layout. It can either have the patch
    // layout discussed in the prev function, or a flat layout if the
    // shape does not meet requirements to be made into patches.
    fn coords(&mut self, shape: &[usize], k: usize) -> Vec<String> {
        if let Some((px, py, [_, pc], [ph, stride])) = self.patch(shape) {
            let (a, b) = (k / pc, k % pc);
            let a = (a / ph) * stride + a % ph;
            let mut out = Vec::new();
            for (i, (corner, offset)) in [(px, a), (py, b)].into_iter().enumerate() {
                let key = format!("patch:{shape:?}:{k}:{i}");
                if let Some(v) = self.lookup(&key) {
                    out.push(v);
                    continue;
                }
                let o = self.int(offset as i64);
                let c = self.op(format!("arith.addi {corner}, {o} : i32"));
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

    // sometimes we get a big tile that is a result of a smaller tile being broadcast.
    // this function returns the coords of the smaller tile that holds the value
    // that belongs to 'coords' position of the bigger tile.
    // input: 'coords', one index per dimension of the big tile, and 'to', the smaller tile's shape.
    // output: one index per dimension of 'to', with None meaning index 0.
    // shapes are lined up from the right: the big tile's extra leading dimensions
    // are dropped, a size-1 dimension of 'to' becomes None, and every other
    // dimension keeps the big tile's index.
    //   e.g. coords (r, c) of a [64, 16] tile:  to [64, 1] -> (r, None),  to [16] -> (c)
    fn project(coords: &[Option<String>], to: &[usize]) -> Vec<Option<String>> {
        let off = coords.len() - to.len();
        to.iter()
            .enumerate()
            .map(|(j, &n)| {
                if n == 1 {
                    None
                } else {
                    coords[off + j].clone()
                }
            })
            .collect()
    }

    // when a smaller tile 'o' is broadcast to shape 'shape'
    // this function, with help of the coords returned by 'project',
    // retruns the value from 'o' that should in the thread's k'th slot of the
    // new shape 'shape'
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

    // when the current thread request for teh element in the position 'proj' of tile 'o'
    // like in the last function, that value could belong to a different thread.
    // so instead of that thread writing to smem for the other thread to read it, it is faster to recompute
    // that value for the thread that needs it.
    fn remat(&mut self, v: usize, coords: &[Option<String>]) -> PyResult<String> {
        let t = self.f.ty(v).clone();
        if t.is_scalar() {
            return Ok(self.slot(v, 0));
        }
        let key = format!("remat:{v}:{coords:?}");
        if let Some(r) = self.lookup(&key) {
            return Ok(r);
        }
        let op = &self.f.ops[*self
            .def
            .get(&v)
            .ok_or_else(|| CompileError::new_err("cannot broadcast a kernel parameter"))?];
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
                let (name, pred) = if e == Elem::F32 {
                    ("arith.cmpf", "olt")
                } else {
                    ("arith.cmpi", "slt")
                };
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
        if n.len() == 1 {
            n[0].clone()
        } else {
            n[r].clone()
        }
    }

    fn tid(&self) -> String {
        "%tid".to_string()
    }

    fn emit_op(&mut self, j: usize) -> PyResult<()> {
        let f = self.f;
        let op = &f.ops[j];
        match op.opcode {
            "program_id" => {
                let axis = match op.attr("axis") {
                    Attr::Int(1) => "y",
                    Attr::Int(2) => "z",
                    _ => "x",
                };
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
                            return Err(CompileError::new_err(
                                "a pointer carried through a loop must stay in the same buffer",
                            ));
                        }
                        for k in 0..a.len() {
                            vals.push(self.slot(*y, k));
                        }
                    }
                    self.line(format!(
                        "scf.yield {} : {}",
                        vals.join(", "),
                        frame.types.join(", ")
                    ));
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
                self.names.insert(r, names.clone());
                if let Some(d) = self.mma_out.get(&r).copied() {
                    let shape = self.f.ty(r).shape.clone();
                    let buf = self.dot_buf[&d].clone();
                    self.mma_to_patch(r, &names, shape[0], shape[1], &buf)?;
                }
            }

            "barrier" => self.line("gpu.barrier".to_string()),
            other => {
                return Err(CompileError::new_err(format!(
                    "MLIR backend: op '{other}' not supported"
                )))
            }
        }
        Ok(())
    }

    fn generate(mut self) -> PyResult<String> {
        let mut args = Vec::new();
        let mut arg_names = Vec::new();
        let zero = "%c0".to_string();
        // bc in metal every kernel argument arrives through a
        // numbered buffer in the GPU memory.
        // so every param needs to be a mlir memref

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

        // record all accumulators
        self.find_accs();

        for j in 0..self.f.ops.len() {
            self.emit_op(j)?;
        }

        let cs: Vec<String> = self
            .f
            .constexprs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let wg = if self.workgroup.is_empty() {
            String::new()
        } else {
            format!(" workgroup({})", self.workgroup.join(", "))
        };
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
        out.extend([
            "    gpu.return".to_string(),
            "  }".to_string(),
            "}".to_string(),
            "}".to_string(),
        ]);
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
        if t.ptr {
            "i32"
        } else {
            mty(t.elem)
        }
    }

    fn op_loop_begin(&mut self, op: &Op) -> PyResult<()> {
        let (lo, hi, st, iv) = (
            op.operands[0],
            op.operands[1],
            op.operands[2],
            op.result.unwrap(),
        );
        let (lo, hi, st) = (self.slot(lo, 0), self.slot(hi, 0), self.slot(st, 0));
        let (lo, hi, one) = (self.to_index(&lo), self.to_index(&hi), self.to_index(&st));
        let i = self.fresh();
        let res = self.fresh();
        let mut frame = Frame {
            inits: op.operands[3..].to_vec(),
            args: vec![],
            results: vec![],
            types: vec![],
        };
        let mut inits = Vec::new();
        let mut count = 0;
        let carried = self.acc_loops.get(&iv).cloned().unwrap_or_default();
        for (idx, &init) in op.operands[3..].iter().enumerate() {
            let mats = if carried.contains(&idx) {
                let shape = self.f.ty(init).shape.clone();
                let zero = self.float(0.0);
                let z = self.op(format!("gpu.subgroup_mma_constant_matrix {zero} : {CMAT}"));
                Some(((shape[0] / 16) * (shape[1] / 16), z))
            } else {
                None
            };
            let n = match &mats {
                Some((n, _)) => *n,
                None => self.regs(init)?,
            };
            let ty = if mats.is_some() {
                CMAT
            } else {
                self.slot_type(init)
            };
            let mut args = Vec::new();
            let mut results = Vec::new();
            for k in 0..n {
                let a = format!("{res}_a{count}");
                let v = match &mats {
                    Some((_, z)) => z.clone(),
                    None => self.slot(init, k),
                };
                inits.push(format!("{a} = {v}"));
                frame.types.push(ty.to_string());
                args.push(a);
                results.push(count);
                count += 1;
            }
            frame.args.push(args);
            frame
                .results
                .push(results.iter().map(|c| c.to_string()).collect());
        }
        for r in frame.results.iter_mut() {
            for c in r.iter_mut() {
                *c = if count == 1 {
                    res.clone()
                } else {
                    format!("{res}#{c}")
                };
            }
        }
        if count == 0 {
            self.line(format!("scf.for {i} = {lo} to {hi} step {one} {{"));
        } else {
            let lhs = if count == 1 {
                res.clone()
            } else {
                format!("{res}:{count}")
            };
            self.line(format!(
                "{lhs} = scf.for {i} = {lo} to {hi} step {one} iter_args({}) -> ({}) {{",
                inits.join(", "),
                frame.types.join(", ")
            ));
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
            (Attr::Int(i), Elem::I1) => self.bool(*i),
            (Attr::Int(i), Elem::I32) => self.int(*i),
            (Attr::Int(i), Elem::F32) => self.float(*i as f64),
            (Attr::Float(x), Elem::F32) => self.float(*x),
            (Attr::Float(x), _) => self.int(*x as i64),
            (Attr::Str(_), _) => unreachable!(),
        };
        self.names.insert(r, vec![v]);
    }

    fn op_arange(&mut self, op: &Op) -> PyResult<()> {
        //the tile ir result ssa
        let r = op.result.unwrap();

        let Attr::Int(start) = op.attr("start") else {
            unreachable!()
        };
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

        // TDO : reconcile element dist with arange
        self.each(r, |g, k| g.operand(src, &shape, k))
    }

    fn op_expand_dims(&mut self, op: &Op) -> PyResult<()> {
        let (src, r) = (op.operands[0], op.result.unwrap());
        if self.f.ty(r).ptr {
            let b = self.base[&src].clone();
            self.base.insert(r, b);
        }
        let Attr::Int(axis) = op.attr("axis") else {
            unreachable!()
        };
        let axis = *axis as usize;
        let shape = self.f.ty(r).shape.clone();
        self.each(r, |g, k| {
            let mut cs: Vec<Option<String>> = g.coords(&shape, k).into_iter().map(Some).collect();
            cs.remove(axis);
            g.remat(src, &cs)
        })
    }

    fn each(
        &mut self,
        r: usize,
        mut f: impl FnMut(&mut Self, usize) -> PyResult<String>,
    ) -> PyResult<()> {
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
        if let Some(&d) = self.mma_add.get(&r) {
            let v = self.names[&d].clone();
            self.names.insert(r, v);
            return Ok(());
        }
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
        let (ptr, r) = (op.operands[0], op.result.unwrap());
        let base = self.base[&ptr].clone();
        let e = mty(self.f.ty(r).elem);
        let shape = self.f.ty(r).shape.clone();

        let w = match self.patch(&shape) {
            Some((_, _, [_, pc], _)) if pc == 2 || pc == 4 => pc,
            _ => 1,
        };
        let n = self.regs(r)?;
        if w > 1 {
            if let Some(pa) = self.affine(ptr).filter(|pa| pa.coef[1].1 == Some(1)) {
                let addrs = self.tile_addrs(&shape, &pa, w, n);
                let vals = self.vector_loads(&base, e, w, &addrs);
                self.names.insert(r, vals);
                return Ok(());
            }
        }
        let slots = self.load_slots(&base, e, w, ptr, n);
        self.names.insert(r, slots);
        Ok(())
    }

    fn tile_addrs(&mut self, shape: &[usize], pa: &Aff, w: usize, n: usize) -> Vec<String> {
        let mut addrs = Vec::new();
        for j in 0..n / w {
            let cs = self.coords(shape, w * j);
            let row = self.sc_mul(&(cs[0].clone(), None), &pa.coef[0]);
            let a = self.sc_add(&row, &(cs[1].clone(), None));
            let a = self.sc_add(&a, &pa.base);
            addrs.push(self.to_index(&a.0));
        }
        addrs
    }

    fn vector_loads(&mut self, base: &str, e: &str, w: usize, addrs: &[String]) -> Vec<String> {
        let vty = format!("vector<{w}x{e}>");
        let mut vals = Vec::new();
        for i in addrs {
            let x = self.op(format!("vector.load {base}[{i}] : memref<?x{e}>, {vty}"));
            for it in 0..w {
                let v = self.op(format!("vector.extract {x}[{it}] : {e} from {vty}"));
                self.vec_src.insert(v.clone(), (x.clone(), it));
                vals.push(v);
            }
        }
        vals
    }

    fn declare(&mut self, decl: String) {
        if !self.workgroup.contains(&decl) {
            self.workgroup.push(decl);
        }
    }

    fn load_slots(&mut self, base: &str, e: &str, w: usize, ptr: usize, n: usize) -> Vec<String> {
        let vty = format!("vector<{w}x{e}>");
        let tys = vec![e; w].join(", ");

        let mut slots = Vec::new();
        if w == 1 {
            for k in 0..n {
                let p = self.slot(ptr, k);
                let i = self.to_index(&p);
                slots.push(self.op(format!("memref.load {base}[{i}] : memref<?x{e}>")));
            }
            return slots;
        }
        for j in 0..n / w {
            let (mut ps, mut is) = (Vec::new(), Vec::new());
            for it in 0..w {
                let p = self.slot(ptr, w * j + it);
                is.push(self.to_index(&p));
                ps.push(p);
            }

            let width = self.int(w as i64);
            let rem = self.op(format!("arith.remui {}, {width} : i32", ps[0]));
            let mut ok = self.op(format!("arith.cmpi eq, {rem}, %zero : i32"));
            for it in 1..w {
                let step = self.int(it as i64);
                let want = self.op(format!("arith.addi {}, {step} : i32", ps[0]));
                let same = self.op(format!("arith.cmpi eq, {}, {want} : i32", ps[it]));
                ok = self.op(format!("arith.andi {ok}, {same} : i1"));
            }

            let res = self.fresh();
            self.line(format!("{res}:{w} = scf.if {ok} -> ({tys}) {{"));
            self.depth += 1;
            let x = self.op(format!("vector.load {base}[{}] : memref<?x{e}>, {vty}", is[0]));
            let mut vals = Vec::new();
            for it in 0..w {
                vals.push(self.op(format!("vector.extract {x}[{it}] : {e} from {vty}")));
            }
            self.line(format!("scf.yield {} : {tys}", vals.join(", ")));
            self.depth -= 1;
            self.line("} else {".to_string());
            self.depth += 1;
            let mut vals = Vec::new();
            for it in 0..w {
                vals.push(self.op(format!("memref.load {base}[{}] : memref<?x{e}>", is[it])));
            }
            self.line(format!("scf.yield {} : {tys}", vals.join(", ")));
            self.depth -= 1;
            self.line("}".to_string());

            for it in 0..w {
                slots.push(format!("{res}#{it}"));
            }
        }
        slots
    }

    fn sc_int(&mut self, v: i64) -> Sc {
        if v == 0 {
            return ("%zero".to_string(), Some(0));
        }
        (self.int(v), Some(v))
    }

    fn sc_add(&mut self, a: &Sc, b: &Sc) -> Sc {
        match (a.1, b.1) {
            (Some(0), _) => b.clone(),
            (_, Some(0)) => a.clone(),
            (Some(x), Some(y)) => self.sc_int(x + y),
            _ => (self.op(format!("arith.addi {}, {} : i32", a.0, b.0)), None),
        }
    }

    fn sc_mul(&mut self, a: &Sc, b: &Sc) -> Sc {
        match (a.1, b.1) {
            (Some(0), _) | (_, Some(0)) => self.sc_int(0),
            (Some(1), _) => b.clone(),
            (_, Some(1)) => a.clone(),
            (Some(x), Some(y)) => self.sc_int(x * y),
            _ => (self.op(format!("arith.muli {}, {} : i32", a.0, b.0)), None),
        }
    }

    fn widen(a: Aff, from: &[usize], to: &[usize]) -> Option<Aff> {
        let off = to.len().checked_sub(from.len())?;
        let zero: Sc = ("%zero".to_string(), Some(0));
        let coef = (0..to.len())
            .map(|d| {
                if d >= off && from[d - off] == to[d] {
                    a.coef[d - off].clone()
                } else {
                    zero.clone()
                }
            })
            .collect();
        Some(Aff { base: a.base, coef })
    }

    fn affine(&mut self, v: usize) -> Option<Aff> {
        let f = self.f;
        let t = f.ty(v);
        if !(t.ptr || t.elem == Elem::I32) {
            return None;
        }
        if t.is_scalar() {
            let name = self.names.get(&v)?.first()?.clone();
            let known = if name == "%zero" {
                Some(0)
            } else {
                self.def.get(&v).and_then(|&d| match f.ops[d].opcode {
                    "constant" => match f.ops[d].attr("value") {
                        Attr::Int(i) => Some(*i),
                        _ => None,
                    },
                    _ => None,
                })
            };
            return Some(Aff {
                base: (name, known),
                coef: vec![],
            });
        }
        let shape = &t.shape;
        let op = &f.ops[*self.def.get(&v)?];
        match op.opcode {
            "arange" => {
                let Attr::Int(start) = op.attr("start") else {
                    return None;
                };
                let base = self.sc_int(*start);
                let one = self.sc_int(1);
                Some(Aff {
                    base,
                    coef: vec![one],
                })
            }
            "broadcast" => {
                let src = op.operands[0];
                let a = self.affine(src)?;
                Self::widen(a, &f.ty(src).shape, shape)
            }
            "expand_dims" => {
                let Attr::Int(axis) = op.attr("axis") else {
                    return None;
                };
                let mut a = self.affine(op.operands[0])?;
                a.coef
                    .insert(*axis as usize, ("%zero".to_string(), Some(0)));
                Some(a)
            }
            "addptr" | "binary" => {
                let kind = if op.opcode == "addptr" {
                    "add"
                } else {
                    let Attr::Str(k) = op.attr("kind") else {
                        return None;
                    };
                    k
                };
                let (x, y) = (op.operands[0], op.operands[1]);
                let a = self.affine(x)?;
                let a = Self::widen(a, &f.ty(x).shape, shape)?;
                let b = self.affine(y)?;
                let b = Self::widen(b, &f.ty(y).shape, shape)?;
                let uniform = |s: &Aff| s.coef.iter().all(|c| c.1 == Some(0));
                match kind {
                    "add" => {
                        let base = self.sc_add(&a.base, &b.base);
                        let mut coef = Vec::new();
                        for (p, q) in a.coef.iter().zip(&b.coef) {
                            coef.push(self.sc_add(p, q));
                        }
                        Some(Aff { base, coef })
                    }
                    "mul" => {
                        let (s, k) = if uniform(&b) {
                            (a, b.base)
                        } else if uniform(&a) {
                            (b, a.base)
                        } else {
                            return None;
                        };
                        let base = self.sc_mul(&s.base, &k);
                        let mut coef = Vec::new();
                        for c in &s.coef {
                            coef.push(self.sc_mul(c, &k));
                        }
                        Some(Aff { base, coef })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
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

    fn stage(
        &mut self,
        v: usize,
        buf: &str,
        wty: &str,
        transpose: bool,
        off: usize,
    ) -> PyResult<()> {
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
        let w = if transpose || owner.is_some() {
            1
        } else {
            match self.patch(&shape) {
                Some((_, _, [_, pc], _)) if (pc == 2 || pc == 4) && shape[1] % pc == 0 && off % pc == 0 => pc,
                _ => 1,
            }
        };
        if w > 1 {
            let vty = format!("vector<{w}xf32>");
            let cols = self.int(shape[1] as i64);
            let mut scalar = Vec::new();
            for k0 in (0..self.regs(v)?).step_by(w) {
                let src: Vec<Option<(String, usize)>> =
                    (0..w).map(|t| self.vec_src.get(&self.slot(v, k0 + t)).cloned()).collect();
                let whole = match &src[0] {
                    Some((x, 0)) => src.iter().enumerate().all(|(t, s)| s.as_ref() == Some(&(x.clone(), t))),
                    _ => false,
                };
                if !whole {
                    scalar.extend(k0..k0 + w);
                    continue;
                }
                let x = src[0].as_ref().unwrap().0.clone();
                let cs = self.coords(&shape, k0);
                let r = self.op(format!("arith.muli {}, {cols} : i32", cs[0]));
                let e = self.op(format!("arith.addi {r}, {} : i32", cs[1]));
                let e = if off > 0 {
                    let o = self.int(off as i64);
                    self.op(format!("arith.addi {e}, {o} : i32"))
                } else {
                    e
                };
                let i = self.to_index(&e);
                self.line(format!("vector.store {x}, {buf}[{i}] : {wty}, {vty}"));
            }
            for k in scalar {
                let cs = self.coords(&shape, k);
                let r = self.op(format!("arith.muli {}, {cols} : i32", cs[0]));
                let e = self.op(format!("arith.addi {r}, {} : i32", cs[1]));
                let e = if off > 0 {
                    let o = self.int(off as i64);
                    self.op(format!("arith.addi {e}, {o} : i32"))
                } else {
                    e
                };
                let i = self.to_index(&e);
                let x = self.slot(v, k);
                self.line(format!("memref.store {x}, {buf}[{i}] : {wty}"));
            }
            return Ok(());
        }
        for k in 0..self.regs(v)? {
            let e = if transpose {
                let cs = self.coords(&shape, k);
                let h = self.int(shape[0] as i64);
                let c = self.op(format!("arith.muli {}, {h} : i32", cs[1]));
                self.op(format!("arith.addi {c}, {} : i32", cs[0]))
            } else if self.patch(&shape).is_some() {
                let cs = self.coords(&shape, k);
                let w = self.int(shape[1] as i64);
                let r = self.op(format!("arith.muli {}, {w} : i32", cs[0]));
                self.op(format!("arith.addi {r}, {} : i32", cs[1]))
            } else {
                self.flat(numel, k)
            };
            let e = if off > 0 {
                let o = self.int(off as i64);
                self.op(format!("arith.addi {e}, {o} : i32"))
            } else {
                e
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

    fn mma_ok(&self, bm: usize, bk: usize, bn: usize) -> bool {
        self.threads == 4 * SIMD_WIDTH && bm % 16 == 0 && bn % 16 == 0 && bk % 8 == 0
    }

    fn find_accs(&mut self) {
        let ops = &self.f.ops;
        let mut users: HashMap<usize, Vec<usize>> = HashMap::new();

        // building a reverse lookup
        // for every value in code, which ops use it
        for (j, op) in ops.iter().enumerate() {
            for &o in &op.operands {
                users.entry(o).or_default().push(j);
            }
        }

        let mut stack: Vec<usize> = Vec::new();
        let mut arg_of: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut end_of: HashMap<usize, usize> = HashMap::new();
        let mut dots: Vec<(usize, usize)> = Vec::new();

        // keeps track if all the do products in all loops
        // helps create the respective accumulators
        for (j, op) in ops.iter().enumerate() {
            match op.opcode {
                "loop_begin" => stack.push(j),
                "loop_arg" => {
                    if let (Some(&l), Attr::Int(k)) = (stack.last(), op.attr("index")) {
                        arg_of.insert(op.result.unwrap(), (l, *k as usize));
                    }
                }
                "loop_end" => {
                    if let Some(l) = stack.pop() {
                        end_of.insert(l, j);
                    }
                }
                "dot" => {
                    if let Some(&l) = stack.last() {
                        dots.push((j, l));
                    }
                }
                _ => {}
            }
        }

        for (j, l) in dots {
            let dop = &ops[j];
            let d = dop.result.unwrap();
            let (at, bt) = (self.f.ty(dop.operands[0]), self.f.ty(dop.operands[1]));
            let (bm, bk, bn) = (at.shape[0], at.shape[1], bt.shape[1]);
            if !self.mma_ok(bm, bk, bn) {
                continue;
            }

            // checks if the dot product's result has any op users
            let Some(du) = users.get(&d) else { continue };

            // checks if the dot product has exactly one
            // user, which is the acc = acc + d op
            // if there are other users, then the result cannot
            // be stored in the 8x8 simd matrix mult acc layout and must be stored in
            // the per thread 2 4x4 blocks layout all the other ops follow
            if du.len() != 1 {
                continue;
            }

            //checks if the op is in fact acc + d
            let sop = &ops[du[0]];
            if sop.opcode != "binary" || !matches!(sop.attr("kind"), Attr::Str("add")) {
                continue;
            }

            // result of the addition
            let s = sop.result.unwrap();

            // takes the operand of the addition that is not teh dot product result
            let p = if sop.operands[0] == d {
                sop.operands[1]
            } else {
                sop.operands[0]
            };

            // checks if that operand is a loop argument. we need
            // it to be a loop argument to pass it through iterations
            let Some(&(pl, k)) = arg_of.get(&p) else {
                continue;
            };
            if pl != l {
                continue;
            }

            // checks if the result of the addition is an argument for
            // the next iteration
            let Some(&e) = end_of.get(&l) else { continue };
            let su = users.get(&s).cloned().unwrap_or_default();
            if su != vec![e] || ops[e].operands.get(k) != Some(&s) {
                continue;
            }

            // checks if p has other users, if it does it cannout be stored in
            // the 8x8 simd matrix mult acc layout
            let pu = users.get(&p).cloned().unwrap_or_default();
            if pu
                .iter()
                .any(|&u| u != du[0] && ops[u].opcode != "loop_result")
            {
                continue;
            }

            // get the op that defines the accumulator's starting value
            let init = ops[l].operands[3 + k];
            let Some(&ii) = self.def.get(&init) else {
                continue;
            };
            let iop = &ops[ii];

            // we want the accumulator to start at 0
            // and have the right result shape
            let zero = iop.opcode == "broadcast"
                && self.def.get(&iop.operands[0]).map_or(false, |&c| {
                    ops[c].opcode == "constant"
                        && matches!(ops[c].attr("value"), Attr::Float(x) if *x == 0.0)
                });
            if !zero || self.f.ty(init).shape != vec![bm, bn] {
                continue;
            }

            // find the loop result that return accumulator out of the loop
            let res = ops
                .iter()
                .enumerate()
                .skip(e)
                .find(|(_, o)| o.opcode == "loop_result" && o.operands.first() == Some(&p));
            let Some((_, rop)) = res else { continue };
            let iv = ops[l].result.unwrap();

            // recording, so the rest of the code knows
            // value iv is in acc matrix layout and not per thread layout
            self.acc_loops.entry(iv).or_default().push(k);
            self.mma_dot.insert(d, p);
            self.mma_add.insert(s, d);
            self.mma_out.insert(rop.result.unwrap(), d);
        }
    }

    fn mma_corner(&mut self, bm: usize, bn: usize) -> (String, String) {
        let tid = self.tid();
        let w = self.int(SIMD_WIDTH as i64);
        let sg = self.op(format!("arith.divui {tid}, {w} : i32"));
        let two = self.int(2);
        let sr = self.op(format!("arith.divui {sg}, {two} : i32"));
        let sc = self.op(format!("arith.remui {sg}, {two} : i32"));
        let qrc = self.int((bm / 2) as i64);
        let qcc = self.int((bn / 2) as i64);
        let row0 = self.op(format!("arith.muli {sr}, {qrc} : i32"));
        let col0 = self.op(format!("arith.muli {sc}, {qcc} : i32"));
        (row0, col0)
    }

    fn mma_to_patch(
        &mut self,
        r: usize,
        mats: &[String],
        bm: usize,
        bn: usize,
        buf: &str,
    ) -> PyResult<()> {
        let (qr, mi, mj) = (bm / 2, bm / 16, bn / 16);
        let (_, col0) = self.mma_corner(bm, bn);
        let bnc = self.int(bn as i64);
        let qrc = self.int(qr as i64);
        let tid = self.tid();
        let w = self.int(SIMD_WIDTH as i64);
        let sg = self.op(format!("arith.divui {tid}, {w} : i32"));
        let two = self.int(2);
        let half = self.op(format!("arith.divui {sg}, {two} : i32"));

        let shape = vec![bm, bn];
        let wv = match self.patch(&shape) {
            Some((_, _, [_, pc], _)) if pc == 2 || pc == 4 => pc,
            _ => 1,
        };
        let regs = self.regs(r)?;
        let mut idx = Vec::new();
        for k0 in (0..regs).step_by(wv) {
            let cs = self.coords(&shape, k0);
            let row = self.op(format!("arith.remui {}, {qrc} : i32", cs[0]));
            let e = self.op(format!("arith.muli {row}, {bnc} : i32"));
            let e = self.op(format!("arith.addi {e}, {} : i32", cs[1]));
            idx.push(self.to_index(&e));
        }

        let mut slots: Vec<String> = Vec::new();
        for h in 0..2 {
            let hc = self.int(h as i64);
            let mine = self.op(format!("arith.cmpi eq, {half}, {hc} : i32"));
            self.line("gpu.barrier".to_string());
            self.line(format!("scf.if {mine} {{"));
            self.depth += 1;
            for i in 0..mi {
                for j in 0..mj {
                    let roff = self.int((8 * i) as i64);
                    let ci = self.op(format!("arith.muli {roff}, {bnc} : i32"));
                    let coff = self.int((8 * j) as i64);
                    let col = self.op(format!("arith.addi {col0}, {coff} : i32"));
                    let ci = self.op(format!("arith.addi {ci}, {col} : i32"));
                    let ci = self.to_index(&ci);
                    self.line(format!("gpu.subgroup_mma_store_matrix {}, {buf}[{ci}] {{leadDimension = {bn} : index}} : {CMAT}, {}", mats[i * mj + j], self.dot_ty(buf)));
                }
            }
            self.depth -= 1;
            self.line("}".to_string());
            self.line("gpu.barrier".to_string());
            let mut vals = Vec::new();
            for e in &idx {
                if wv == 1 {
                    vals.push(self.op(format!("memref.load {buf}[{e}] : {}", self.dot_ty(buf))));
                } else {
                    let x = self.op(format!(
                        "vector.load {buf}[{e}] : {}, vector<{wv}xf32>",
                        self.dot_ty(buf)
                    ));
                    for t in 0..wv {
                        vals.push(self.op(format!(
                            "vector.extract {x}[{t}] : f32 from vector<{wv}xf32>"
                        )));
                    }
                }
            }
            slots = if h == 0 {
                vals
            } else {
                let mut out = Vec::new();
                for (v1, v0) in vals.iter().zip(&slots) {
                    out.push(self.op(format!("arith.select {mine}, {v1}, {v0} : f32")));
                }
                out
            };
        }
        self.names.insert(r, slots);
        Ok(())
    }

    fn dot_ty(&self, buf: &str) -> String {
        let decl = self
            .workgroup
            .iter()
            .find(|w| w.starts_with(&format!("{buf} :")))
            .unwrap();
        decl[buf.len() + 3..].to_string()
    }

    fn dot_mma(
        &mut self,
        a: usize,
        b: usize,
        r: usize,
        bm: usize,
        bk: usize,
        bn: usize,
        d: usize,
        acc: Option<Vec<String>>,
    ) -> PyResult<()> {

        // create shared memory buffer
        let buf = format!("%dot{d}s");
        let boff = bm * bk;
        let size = (bm * bk + bk * bn).max(bm / 2 * bn);
        let ty = format!("memref<{size}xf32, #gpu.address_space<workgroup>>");
        self.declare(format!("{buf} : {ty}"));
        self.dot_buf.insert(r, buf.clone());

        // stage values to the SMEM buffer
        // we need barrier bc diff simds 
        // could be in diff iterations
        // before writing it is importand all simds
        // complete their reads and before reading
        // it is important all simds complete their writes
        self.line("gpu.barrier".to_string());
        self.stage(a, &buf, &ty, false, 0)?;
        self.stage(b, &buf, &ty, false, boff)?;
        self.line("gpu.barrier".to_string());
        
        // no of 8x8 blocks along both axes
        // for C for one SIMD quadrant
        let (mi, mj) = (bm / 16, bn / 16);

        // total no of accumulators for one SIMD quad
        let n = mi * mj;

        // top left corner of this SIMD qudrant in C
        let (row0, col0) = self.mma_corner(bm, bn);


        let bkc = self.int(bk as i64);
        let bnc = self.int(bn as i64);
        let mut arows = Vec::new();

        // to calculate all 8x8 blocks of current qudrant
        // which are the rows of A required
        for i in 0..mi {
            let off = self.int((8 * i) as i64);
            let row = self.op(format!("arith.addi {row0}, {off} : i32"));
            arows.push(self.op(format!("arith.muli {row}, {bkc} : i32")));
        }

        let mut bcols = Vec::new();

        // to calculate all 8x8 blocks of current qudrant
        // which are the cols of B required
        for j in 0..mj {
            let off = self.int((8 * j + boff) as i64);
            bcols.push(self.op(format!("arith.addi {col0}, {off} : i32")));
        }

        // initialize all the n accumulators needed for the simd
        let start: Vec<String> = match &acc {
            Some(v) => v.clone(),
            None => {
                let zero = self.float(0.0);
                let z = self.op(format!("gpu.subgroup_mma_constant_matrix {zero} : {CMAT}"));
                vec![z; n]
            }
        };

        let (lo, hi, step) = (self.idx(0), self.idx(bk as i64), self.idx(8));
        let res = self.fresh();
        let kk = self.fresh();
        let accs: Vec<String> = (0..n).map(|t| format!("{res}_m{t}")).collect();
        let inits: Vec<String> = accs
            .iter()
            .zip(&start)
            .map(|(x, z)| format!("{x} = {z}"))
            .collect();
        let types = vec![CMAT; n].join(", ");
        let lhs = if n == 1 {
            res.clone()
        } else {
            format!("{res}:{n}")
        };
        self.line(format!(
            "{lhs} = scf.for {kk} = {lo} to {hi} step {step} iter_args({}) -> ({types}) {{",
            inits.join(", ")
        ));
        self.depth += 1;
        let kki = self.op(format!("arith.index_cast {kk} : index to i32"));
        
        
        let krow = self.op(format!("arith.muli {kki}, {bnc} : i32"));
        let mut bv = Vec::new();
        for j in 0..mj {
            let bi = self.op(format!("arith.addi {krow}, {} : i32", bcols[j]));
            let bi = self.to_index(&bi);
            bv.push(self.op(format!("gpu.subgroup_mma_load_matrix {buf}[{bi}] {{leadDimension = {bn} : index}} : {ty} -> {BMAT}")));
        }
        let mut next = Vec::new();
        for i in 0..mi {
            let ai = self.op(format!("arith.addi {}, {kki} : i32", arows[i]));
            let ai = self.to_index(&ai);
            let av = self.op(format!("gpu.subgroup_mma_load_matrix {buf}[{ai}] {{leadDimension = {bk} : index}} : {ty} -> {AMAT}"));
            for j in 0..mj {
                next.push(self.op(format!(
                    "gpu.subgroup_mma_compute {}, {}, {} : {AMAT}, {BMAT} -> {CMAT}",
                    av,
                    bv[j],
                    accs[i * mj + j]
                )));
            }
        }
        self.line(format!("scf.yield {} : {types}", next.join(", ")));
        self.depth -= 1;
        self.line("}".to_string());

        let mats: Vec<String> = (0..n)
            .map(|t| {
                if n == 1 {
                    res.clone()
                } else {
                    format!("{res}#{t}")
                }
            })
            .collect();
        if acc.is_some() {
            self.names.insert(r, mats);
            return Ok(());
        }
        self.mma_to_patch(r, &mats, bm, bn, &buf)
    }

    // op_dot: C[bm, bn] = A[bm, bk] · B[bk, bn] for one tile, using 8×8 simdgroup matrices.
    //
    // Example: bm = bn = 64, bk = 16, 128 threads = 4 SIMD-groups × 32 lanes.
    //
    // ── Shapes ─────────────────────────────────────────────────────────────────────
    // mma_ok(bm, bk, bn) is true when the tile can be cut into whole 8×8 hardware matrices:
    //   threads == 128   C is split 2×2 into quadrants, one per SIMD-group
    //   bm % 16 == 0     each quadrant (bm/2 rows) is whole 8-row blocks
    //   bn % 16 == 0     each quadrant (bn/2 cols) is whole 8-col blocks
    //   bk % 8  == 0     the K loop takes 8 columns of A at a time
    // true  → dot_mma
    // false → compile error: the matrix path is the only one for now
    //
    // ── find_accs (runs once, before code generation) ───────────────────────────────
    // Looks for   acc = zeros([bm, bn]); loop { acc = acc + dot(a, b) }; use(acc)
    // When it matches, acc is kept as 16 matrices across loop iterations:
    //   acc_loops[iv] = [k]   op_loop_begin carries loop slot k as matrices, starting at 0.0
    //   mma_dot[d]    = p     this dot starts from p's matrices instead of zero (`acc` below)
    //   mma_add[s]    = d     op_binary skips the add; s is d's matrices
    //   mma_out[res]  = d     loop_result converts to per-thread values once (mma_to_patch)
    // Without the match, every iteration would convert matrices → per-thread values and
    // back: 4 extra barriers and 32 KB of shared-memory traffic per iteration.
    //
    // ── dot_mma ─────────────────────────────────────────────────────────────────────
    // 1. One shared buffer %dot{d}s of max(bm·bk + bk·bn, bm/2·bn) floats (8 KB here):
    //        [ A row-major bm×bk | B row-major bk×bn ]   offsets 0 and boff = bm·bk
    //    later reused by mma_to_patch to hold half of C (bm/2 × bn).
    // 2. barrier → stage A, stage B → barrier
    //    the first barrier: nobody may still be reading the old contents (write after read)
    //    the second: every value is written before anyone reads it (read after write)
    // 3. mma_corner gives each SIMD-group its quadrant:
    //        sg = tid/32, row0 = (sg/2)·bm/2, col0 = (sg%2)·bn/2
    //            ┌── sg0 ──┬── sg1 ──┐
    //            ├── sg2 ──┼── sg3 ──┤     each 32×32 quadrant = 4×4 blocks of 8×8
    //            └─────────┴─────────┘     → mi·mj = 16 accumulator matrices C[i·mj + j]
    // 4. for kk in 0..bk step 8:
    //        av[i] = load A block at (row0+8i)·bk + kk          leadDimension = bk
    //        bv[j] = load B block at kk·bn + col0 + 8j + boff   leadDimension = bn
    //        C[i·mj + j] += av[i] × bv[j]                       for all i, j
    //    each A block is reused mj times and each B block mi times
    // 5. if acc was carried (find_accs), the 16 matrices are the result;
    //    otherwise mma_to_patch converts them now.
    //
    // ── mma_to_patch (matrices → patch layout) ──────────────────────────────────────
    // Which lane holds which value inside a matrix is hidden, but store and the other ops
    // need each thread to hold known (row, col) positions (patch layout). The buffer holds
    // only bm/2 rows, so the conversion takes two passes:
    //   pass h: barrier; SIMD-groups with sg/2 == h store their matrices; barrier;
    //           every thread reads its positions at (row % (bm/2))·bn + col (float4 when pc = 4)
    //   then arith.select keeps the pass whose half holds the thread's rows
    // dot_ty(buf) looks up the buffer's memref type for printing these ops.
    fn op_dot(&mut self, op: &Op) -> PyResult<()> {
        let (a, b, r) = (op.operands[0], op.operands[1], op.result.unwrap());
        let (bm, bk) = (self.f.ty(a).shape[0], self.f.ty(a).shape[1]);
        let bn = self.f.ty(b).shape[1];
        let d = self.n_dot;
        self.n_dot += 1;
        if self.mma_ok(bm, bk, bn) {
            let acc = self.mma_dot.get(&r).map(|p| self.names[p].clone());
            return self.dot_mma(a, b, r, bm, bk, bn, d, acc);
        } else {
            return Err(CompileError::new_err(format!(
                "tl.dot: only the simdgroup_matrix path is supported for now; it needs 128 threads, \
                 bm and bn multiples of 16 and bk a multiple of 8, got [{bm}, {bk}] x [{bk}, {bn}]"
            )));
        }
    }
}

pub fn generate(f: &Function, num_warps: usize) -> PyResult<String> {
    Gen {
        f,
        threads: num_warps * SIMD_WIDTH,
        body: vec![],
        workgroup: vec![],
        names: HashMap::new(),
        base: HashMap::new(),
        tmp: 0,
        depth: 0,
        loops: vec![],
        done: vec![],
        def: HashMap::new(),
        cache: vec![HashMap::new()],
        n_dot: 0,
        acc_loops: HashMap::new(),
        mma_dot: HashMap::new(),
        mma_add: HashMap::new(),
        mma_out: HashMap::new(),
        dot_buf: HashMap::new(),
        vec_src: HashMap::new(),
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
        return Err(CompileError::new_err(format!(
            "{tool} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let Some((opt, msl)) = text.split_once("//--MSL--\n") else {
        return Err(CompileError::new_err(format!("{tool} printed no Metal")));
    };
    Ok((opt.to_string(), msl.to_string()))
}
