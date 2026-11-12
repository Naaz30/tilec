use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyTuple};

use crate::ir::{Attr, Elem, Function, Type};
use crate::CompileError;

const MAX_UNROLL: i64 = 256;

#[derive(Clone)]
pub enum Val<'py> {
    Int(i64),               
    Float(f64),            
    Obj(Bound<'py, PyAny>), 
    V(usize),
}

pub struct FrontEnd<'py> {
    pub func: Function,
    env: HashMap<String, Val<'py>>,
    globals: Bound<'py, PyDict>,
    src_lines: Vec<String>,
    first_line: usize,
    loop_outer: Vec<HashSet<String>>,     
}

fn kind(node: &Bound<'_, PyAny>) -> PyResult<String> {
    node.getattr("__class__")?.getattr("__name__")?.extract()
}

fn builtin_name(obj: &Bound<'_, PyAny>) -> Option<String> {
    obj.getattr("tilec_builtin").ok()?.extract().ok()
}

fn py_to_val<'py>(obj: Bound<'py, PyAny>) -> PyResult<Val<'py>> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(Val::Obj(obj));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Val::Int(obj.extract()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Val::Float(obj.extract()?));
    }
    Ok(Val::Obj(obj))
}

fn assigned_names(body: &Bound<'_, PyAny>, out: &mut Vec<String>) -> PyResult<()> {
    for stmt in body.try_iter()? {
        let stmt = stmt?;
        match kind(&stmt)?.as_str() {
            "Assign" => {
                let target = stmt.getattr("targets")?.get_item(0)?;
                if kind(&target)? == "Name" {
                    let name: String = target.getattr("id")?.extract()?;
                    if !out.contains(&name) {
                        out.push(name);
                    }
                }
            }
            "For" => assigned_names(&stmt.getattr("body")?, out)?,
            _ => {}
        }
    }
    Ok(())
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if x == y || y == 1 { x } else if x == 1 { y } else { return None };
    }
    Some(out)
}

impl<'py> FrontEnd<'py> {
    pub fn new(name: &str, globals: Bound<'py, PyDict>, source: &str, first_line: usize) -> Self {
        FrontEnd {
            func: Function::new(name),
            env: HashMap::new(),
            globals,
            src_lines: source.lines().map(String::from).collect(),
            first_line,
            loop_outer: Vec::new(),        
        }
    }

    pub fn build(&mut self, fdef: &Bound<'py, PyAny>, param_types: &HashMap<String, Type>,
                 constexprs: &Bound<'py, PyDict>) -> PyResult<()> {
        for arg in fdef.getattr("args")?.getattr("args")?.try_iter()? {
            let name: String = arg?.getattr("arg")?.extract()?;
            if let Some(value) = constexprs.get_item(&name)? {
                self.func.constexprs.push((name.clone(), value.repr()?.extract()?));
                self.env.insert(name, py_to_val(value)?);
            } else {
                let ty = param_types.get(&name)
                    .ok_or_else(|| CompileError::new_err(format!("no type for parameter '{name}'")))?;
                let v = self.func.add_param(&name, ty.clone());
                self.env.insert(name, Val::V(v));
            }
        }
        for stmt in fdef.getattr("body")?.try_iter()? {
            self.visit_stmt(&stmt?)?;
        }
        Ok(())
    }

    fn error(&self, node: &Bound<'py, PyAny>, msg: &str) -> PyErr {
        let line: usize = node.getattr("lineno").and_then(|l| l.extract()).unwrap_or(1);
        let text = self.src_lines.get(line - 1).map(|s| s.trim()).unwrap_or("");
        CompileError::new_err(format!("{msg}\n  at line {}: {text}", line + self.first_line - 1))
    }

    fn visit_stmt(&mut self, node: &Bound<'py, PyAny>) -> PyResult<()> {
        match kind(node)?.as_str() {
            "Assign" => {
                let target = node.getattr("targets")?.get_item(0)?;
                if kind(&target)? != "Name" {
                    return Err(self.error(node, "only assignment to a name is supported"));
                }
                let value = self.eval(&node.getattr("value")?)?;
                self.env.insert(target.getattr("id")?.extract()?, value);
            }
            "Expr" => {
                let value = node.getattr("value")?;
                if kind(&value)? != "Constant" {      
                    self.eval(&value)?;
                }
            }
            "For" => self.visit_for(node)?,
            other => return Err(self.error(node, &format!("unsupported statement: {other}"))),
        }
        Ok(())
    }

    fn visit_for(&mut self, node: &Bound<'py, PyAny>) -> PyResult<()> {
        let it = node.getattr("iter")?;
        
        if kind(&it)? == "Call" {
            let func = it.getattr("func")?;
            if kind(&func)? == "Name" && func.getattr("id")?.extract::<String>()? == "range" {
               return self.visit_runtime_for(node, &it);
            }
        }
        let n = match self.eval(&it.getattr("args")?.get_item(0)?)? {
            Val::Int(n) => n,
            _ => return Err(self.error(node, "tl.static_range needs a compile-time count")),
        };
        if n > MAX_UNROLL {
            return Err(self.error(node, &format!("loop of {n} iterations is too long to unroll")));
        }
        let target: String = node.getattr("target")?.getattr("id")?.extract()?;
        for i in 0..n {                               
            self.env.insert(target.clone(), Val::Int(i));
            for stmt in node.getattr("body")?.try_iter()? {
                self.visit_stmt(&stmt?)?;
            }
        }
        Ok(())
    }

    fn visit_runtime_for(&mut self, node: &Bound<'py, PyAny>, it: &Bound<'py, PyAny>) -> PyResult<()> {
        let args = it.getattr("args")?;
        let mut bounds = Vec::new();
        for a in args.try_iter()? {
            bounds.push(self.eval(&a?)?);
        }
        let (start, stop, step) = match bounds.len() {
            1 => (Val::Int(0), bounds[0].clone(), Val::Int(1)),
            2 => (bounds[0].clone(), bounds[1].clone(), Val::Int(1)),
            3 => (bounds[0].clone(), bounds[1].clone(), bounds[2].clone()),
            _ => return Err(self.error(node, "range takes 1 to 3 arguments")),
        };
        if let Val::Int(s) = step {
            if s <= 0 {
                return Err(self.error(node, "range step must be positive"));
            }
        }
        let start = self.to_value(start, None, node)?;
        let end = self.to_value(stop, None, node)?;
        let step = self.to_value(step, None, node)?;
        for v in [start, end, step] {
            if *self.func.ty(v) != Type::scalar(Elem::I32) {
                return Err(self.error(node, "range bounds must be integers"));
            }
        }
        let target: String = node.getattr("target")?.getattr("id")?.extract()?;

        let mut assigned = Vec::new();
        assigned_names(&node.getattr("body")?, &mut assigned)?;
        let carried: Vec<String> = assigned.into_iter()
            .filter(|n| *n != target && self.env.contains_key(n))
            .collect();
        let mut inits = Vec::new();
        for name in &carried {
            let init = match self.env[name].clone() {
                Val::V(v) => v,
                Val::Obj(_) => return Err(self.error(node, &format!("'{name}' cannot be reassigned inside a runtime loop"))),
                c => self.to_value(c, None, node)?,
            };
            inits.push(init);
        }

        let mut operands = vec![start, end, step];
        operands.extend(&inits);
        let iv = self.func.emit("loop_begin", operands, Some(Type::scalar(Elem::I32)), vec![]).unwrap();
        let saved = self.env.clone();
        self.loop_outer.push(saved.keys().cloned().collect());
        let mut loop_args = Vec::new();
        for (k, (name, init)) in carried.iter().zip(&inits).enumerate() {
            let ty = self.func.ty(*init).clone();
            let a = self.func.emit("loop_arg", vec![], Some(ty), vec![("index", Attr::Int(k as i64))]).unwrap();
            self.env.insert(name.clone(), Val::V(a));
            loop_args.push(a);
        }
        self.env.insert(target, Val::V(iv));
        for stmt in node.getattr("body")?.try_iter()? {
            self.visit_stmt(&stmt?)?;
        }

        let mut yields = Vec::new();
        for (name, a) in carried.iter().zip(&loop_args) {
            let new = match self.env[name].clone() {
                Val::V(v) => v,
                Val::Obj(_) => return Err(self.error(node, &format!("'{name}' cannot be reassigned inside a runtime loop"))),
                c => self.to_value(c, Some(*a), node)?,
            };
            let (before, after) = (self.func.ty(*a).clone(), self.func.ty(new).clone());
            if before != after {
                return Err(self.error(node, &format!(
                    "'{name}' changes type inside the loop ({before} before, {after} after); give it its final shape before the loop")));
            }
            yields.push(new);
        }
        self.loop_outer.pop();
        self.func.emit("loop_end", yields, None, vec![]);
        self.env = saved;
        for (k, (name, a)) in carried.iter().zip(&loop_args).enumerate() {
            let ty = self.func.ty(*a).clone();
            let r = self.func.emit("loop_result", vec![*a], Some(ty), vec![("index", Attr::Int(k as i64))]).unwrap();
            self.env.insert(name.clone(), Val::V(r));
        }
        Ok(())
    }

   
    fn eval(&mut self, node: &Bound<'py, PyAny>) -> PyResult<Val<'py>> {
        match kind(node)?.as_str() {
            "Constant" => py_to_val(node.getattr("value")?),
            "Name" => {
                let id: String = node.getattr("id")?.extract()?;
                if let Some(v) = self.env.get(&id) {
                    return Ok(v.clone());
                }
                match self.globals.get_item(&id)? {
                    Some(obj) => py_to_val(obj),
                    None => Err(self.error(node, &format!("name '{id}' is not defined"))),
                }
            }
            "Attribute" => {                         
                let attr: String = node.getattr("attr")?.extract()?;
                let value = node.getattr("value")?;
                match self.eval(&value)? {
                    Val::Obj(base) => match base.getattr(attr.as_str()) {
                        Ok(obj) => py_to_val(obj),
                        Err(_) => {
                           
                            let what: String = match value.getattr("id") {
                                Ok(id) => id.extract()?,
                                Err(_) => base.repr()?.extract()?,
                            };
                            Err(self.error(node, &format!("{what} has no attribute '{attr}'")))
                        }
                    },
                    _ => Err(self.error(node, &format!("tensors have no attribute '{attr}'"))),
                }
            }
            "BinOp" => {
                let op = match kind(&node.getattr("op")?)?.as_str() {
                    "Add" => "add",
                    "Mult" => "mul",
                    "BitAnd" => "and",
                    other => return Err(self.error(node, &format!("unsupported operator {other}"))),
                };
                let a = self.eval(&node.getattr("left")?)?;
                let b = self.eval(&node.getattr("right")?)?;
                self.binary(op, a, b, node)
            }
            "Compare" => {
                let ops = node.getattr("ops")?;
                if ops.len()? != 1 || kind(&ops.get_item(0)?)? != "Lt" {
                    return Err(self.error(node, "only a single `<` comparison is supported"));
                }
                let a = self.eval(&node.getattr("left")?)?;
                let b = self.eval(&node.getattr("comparators")?.get_item(0)?)?;
                self.compare("lt", a, b, node)
            }
            "Call" => self.eval_call(node),
            "Tuple" => {
                let mut items = Vec::new();
                for e in node.getattr("elts")?.try_iter()? {
                    match self.eval(&e?)? {
                        Val::Int(i) => items.push(i),
                        _ => return Err(self.error(node, "only tuples of compile-time integers are supported")),
                    }
                }
                Ok(Val::Obj(PyTuple::new(node.py(), items)?.into_any()))
            }
            "Subscript" => self.subscript(node),
            other => Err(self.error(node, &format!("unsupported expression: {other}"))),
        }
    }

    fn eval_call(&mut self, node: &Bound<'py, PyAny>) -> PyResult<Val<'py>> {
        let name = match self.eval(&node.getattr("func")?)? {
            Val::Obj(f) => builtin_name(&f),
            _ => None,
        };
        let Some(name) = name else {
            return Err(self.error(node, "only tl builtins can be called inside a kernel"));
        };
        let mut args = Vec::new();
        for a in node.getattr("args")?.try_iter()? {
            args.push(self.eval(&a?)?);
        }
        let mut kwargs = HashMap::new();
        for kw in node.getattr("keywords")?.try_iter()? {
            let kw = kw?;
            kwargs.insert(kw.getattr("arg")?.extract::<String>()?, self.eval(&kw.getattr("value")?)?);
        }
       
        let mut arg = |i: usize, key: &str| -> Option<Val<'py>> {
            args.get(i).cloned().or_else(|| kwargs.remove(key))
        };
        match name.as_str() {
            "program_id" => {
                let axis = match arg(0, "axis") { Some(Val::Int(a)) => a, None => 0, _ => -1 };
                if !(0..3).contains(&axis) {
                    return Err(self.error(node, "tl.program_id takes axis 0, 1 or 2"));
                }
                Ok(self.v("program_id", vec![], Type::scalar(Elem::I32), vec![("axis", Attr::Int(axis))]))
            }
            "zeros" => {
                let shape: Vec<usize> = match arg(0, "shape") {
                    Some(Val::Int(n)) if n > 0 => vec![n as usize],
                    Some(Val::Obj(t)) => t.extract::<Vec<i64>>()
                        .map_err(|_| self.error(node, "tl.zeros needs a compile-time shape"))?
                        .into_iter().map(|n| n as usize).collect(),
                    _ => return Err(self.error(node, "tl.zeros needs a compile-time shape")),
                };
                if shape.iter().any(|&n| n == 0 || n & (n - 1) != 0) {
                    return Err(self.error(node, &format!("tl.zeros dimensions must be powers of two (got {shape:?})")));
                }
                let z = self.func.emit("constant", vec![], Some(Type::scalar(Elem::F32)),
                                       vec![("value", Attr::Float(0.0))]).unwrap();
                Ok(Val::V(self.broadcast_to(z, &shape, node)?))
            }
            "dot" => {
                let (Some(Val::V(a)), Some(Val::V(b))) = (arg(0, "a"), arg(1, "b")) else {
                    return Err(self.error(node, "use tl.dot(a, b) with two 2-D tiles"));
                };
                let (ta, tb) = (self.func.ty(a).clone(), self.func.ty(b).clone());
                if ta.shape.len() != 2 || tb.shape.len() != 2 || ta.shape[1] != tb.shape[0] {
                    return Err(self.error(node, &format!("tl.dot needs (M, K) and (K, N) tiles, got {ta} and {tb}")));
                }
                if ta.elem != Elem::F32 || tb.elem != Elem::F32 || ta.ptr || tb.ptr {
                    return Err(self.error(node, "tl.dot needs float tiles"));
                }
                let ty = Type { elem: Elem::F32, ptr: false, shape: vec![ta.shape[0], tb.shape[1]] };
                Ok(self.v("dot", vec![a, b], ty, vec![]))
            }
            "arange" => {
                let (Some(Val::Int(start)), Some(Val::Int(end))) = (arg(0, "start"), arg(1, "end")) else {
                    return Err(self.error(node, "tl.arange needs compile-time start and end"));
                };
                let n = end - start;
                if n <= 0 || n & (n - 1) != 0 {
                    return Err(self.error(node, &format!("arange size must be a positive power of two (got {n})")));
                }
                Ok(self.v("arange", vec![], Type { elem: Elem::I32, ptr: false, shape: vec![n as usize] },
                          vec![("start", Attr::Int(start)), ("end", Attr::Int(end))]))
            }
            "load" => {
                let (Some(Val::V(ptr)), Some(mask), Some(other)) = (arg(0, "pointer"), arg(1, "mask"), arg(2, "other")) else {
                    return Err(self.error(node, "use tl.load(pointer, mask=..., other=...)"));
                };
                let pty = self.func.ty(ptr).clone();
                if !pty.ptr {
                    return Err(self.error(node, "tl.load needs a pointer"));
                }
                let Val::V(mask) = mask else { return Err(self.error(node, "mask must be a tensor")) };
                let other = match other {
                    Val::Int(i) => i as f64,
                    Val::Float(f) => f,
                    _ => return Err(self.error(node, "`other` must be a number")),
                };
                let m = self.broadcast_to(mask, &pty.shape, node)?;
                let o = self.func.emit("constant", vec![], Some(Type::scalar(pty.elem)),
                                       vec![("value", Attr::Float(other))]).unwrap();
                let o = self.broadcast_to(o, &pty.shape, node)?;  
                Ok(self.v("load", vec![ptr, m, o], Type { elem: pty.elem, ptr: false, shape: pty.shape }, vec![]))
            }
            "store" => {
                let (Some(Val::V(ptr)), Some(value)) = (arg(0, "pointer"), arg(1, "value")) else {
                    return Err(self.error(node, "use tl.store(pointer, value, mask=...)"));
                };
                let mask = arg(2, "mask");
                let value = self.to_value(value, None, node)?;
                let (ptr, value) = self.broadcast_pair(ptr, value, node)?;
                let mut operands = vec![ptr, value];
                if let Some(mask) = mask {
                    let Val::V(mask) = mask else { return Err(self.error(node, "mask must be a tensor")) };
                    let shape = self.func.ty(ptr).shape.clone();
                    operands.push(self.broadcast_to(mask, &shape, node)?);
                }
                self.func.emit("store", operands, None, vec![]);
                Ok(Val::Obj(node.py().None().into_bound(node.py())))
            }
            "sum" => {
                let Some(Val::V(x)) = arg(0, "input") else {
                    return Err(self.error(node, "tl.sum needs a tensor"));
                };
                let elem = self.func.ty(x).elem;
                Ok(self.v("reduce", vec![x], Type::scalar(elem), vec![("kind", Attr::Str("sum"))]))
            }
            "mul" => {
                let (Some(a), Some(b)) = (arg(0, "x"), arg(1, "y")) else {
                   return Err(self.error(node, "use tl.mul(x, y)"));
                };
                self.binary("mul", a, b, node)
            }
            "barrier" => {
                self.func.emit("barrier", vec![], None, vec![]);
                Ok(Val::Obj(node.py().None().into_bound(node.py())))
            }
            "static_range" => Err(self.error(node, "tl.static_range can only be used as a for-loop iterator")),
            other => Err(self.error(node, &format!("tl.{other} is not supported"))),
        }
    }

    fn v(&mut self, opcode: &'static str, operands: Vec<usize>, ty: Type,
         attrs: Vec<(&'static str, Attr)>) -> Val<'py> {
        Val::V(self.func.emit(opcode, operands, Some(ty), attrs).unwrap())
    }

    fn to_value(&mut self, x: Val<'py>, like: Option<usize>, node: &Bound<'py, PyAny>) -> PyResult<usize> {
        let like_elem = like.map(|l| self.func.ty(l).clone()).filter(|t| !t.ptr).map(|t| t.elem);
        let (elem, value) = match x {
            Val::V(v) => return Ok(v),
            Val::Int(i) => {
                let e = like_elem.unwrap_or(Elem::I32);
                (e, if e.is_int() { Attr::Int(i) } else { Attr::Float(i as f64) })
            }
            Val::Float(f) => {
                let e = like_elem.unwrap_or(Elem::F32);
                (e, if e.is_int() { Attr::Int(f as i64) } else { Attr::Float(f) })
            }
            Val::Obj(o) => return Err(self.error(node, &format!("expected a number or tensor, got {}", o.repr()?))),
        };
        Ok(self.func.emit("constant", vec![], Some(Type::scalar(elem)), vec![("value", value)]).unwrap())
    }

    fn broadcast_to(&mut self, v: usize, shape: &[usize], node: &Bound<'py, PyAny>) -> PyResult<usize> {
        let ty = self.func.ty(v).clone();
        if ty.shape == shape {
            return Ok(v);                           
        }
        if broadcast_shapes(&ty.shape, shape).as_deref() != Some(shape) {
            return Err(self.error(node, &format!("cannot broadcast {:?} to {:?}", ty.shape, shape)));
        }
        Ok(self.func.emit("broadcast", vec![v], Some(ty.with_shape(shape)), vec![]).unwrap())
    }

    fn broadcast_pair(&mut self, a: usize, b: usize, node: &Bound<'py, PyAny>) -> PyResult<(usize, usize)> {
        let (sa, sb) = (self.func.ty(a).shape.clone(), self.func.ty(b).shape.clone());
        let Some(shape) = broadcast_shapes(&sa, &sb) else {
            return Err(self.error(node, &format!("shapes {sa:?} and {sb:?} do not broadcast")));
        };
        Ok((self.broadcast_to(a, &shape, node)?, self.broadcast_to(b, &shape, node)?))
    }

    fn binary(&mut self, op: &'static str, a: Val<'py>, b: Val<'py>, node: &Bound<'py, PyAny>) -> PyResult<Val<'py>> {
        let num = |v: &Val| match v { Val::Int(i) => Some(*i as f64), Val::Float(f) => Some(*f), _ => None };
        match (&a, &b) {
            (Val::Int(x), Val::Int(y)) => return Ok(Val::Int(match op { "add" => x + y, "mul" => x * y, _ => x & y })),
            _ if op == "and" && num(&a).is_some() && num(&b).is_some() => {
                return Err(self.error(node, "& needs integers or masks"));
            }
            _ if num(&a).is_some() && num(&b).is_some() => {
                let (x, y) = (num(&a).unwrap(), num(&b).unwrap());
                return Ok(Val::Float(if op == "add" { x + y } else { x * y }));
            }
            _ => {}
        }
    
        if let Val::V(p) = a {
            if self.func.ty(p).ptr {
                if op != "add" {
                    return Err(self.error(node, "pointers only support +"));
                }
                let off = self.to_value(b, None, node)?;
                let (p, off) = self.broadcast_pair(p, off, node)?;
                let ty = self.func.ty(p).clone();
                return Ok(self.v("addptr", vec![p, off], ty, vec![]));
            }
        }
        let like_b = if let Val::V(v) = b { Some(v) } else { None };
        let a = self.to_value(a, like_b, node)?;
        let b = self.to_value(b, Some(a), node)?;
        if self.func.ty(a).elem != self.func.ty(b).elem {
            return Err(self.error(node, &format!("mixed types {} and {}", self.func.ty(a), self.func.ty(b))));
        }
        if op == "and" && self.func.ty(a).elem == Elem::F32 {
            return Err(self.error(node, "& needs integers or masks"));
        }
        let (a, b) = self.broadcast_pair(a, b, node)?;
        let ty = self.func.ty(a).clone();
        Ok(self.v("binary", vec![a, b], ty, vec![("kind", Attr::Str(op))]))
    }

    fn subscript(&mut self, node: &Bound<'py, PyAny>) -> PyResult<Val<'py>> {
        let Val::V(mut x) = self.eval(&node.getattr("value")?)? else {
            return Err(self.error(node, "only tiles can be indexed"));
        };
        let index = node.getattr("slice")?;
        let parts: Vec<Bound<'py, PyAny>> = if kind(&index)? == "Tuple" {
            index.getattr("elts")?.try_iter()?.collect::<PyResult<_>>()?
        } else {
            vec![index]
        };
        let rank = self.func.ty(x).shape.len();
        let mut dim = 0;
        let mut kept = 0;
        for p in &parts {
            match kind(p)?.as_str() {
                "Slice" if p.getattr("lower")?.is_none() && p.getattr("upper")?.is_none()
                           && p.getattr("step")?.is_none() => {
                    dim += 1;
                    kept += 1;
                }
                "Constant" if p.getattr("value")?.is_none() => {
                    let mut ty = self.func.ty(x).clone();
                    ty.shape.insert(dim, 1);
                    x = self.func.emit("expand_dims", vec![x], Some(ty), vec![("axis", Attr::Int(dim as i64))]).unwrap();
                    dim += 1;
                }
                _ => return Err(self.error(node, "only x[:, None] and x[None, :] style indexing is supported")),
            }
        }
        if kept != rank {
            return Err(self.error(node, &format!("indexing a {rank}-D tile needs {rank} `:`")));
        }
        Ok(Val::V(x))
    }

    fn compare(&mut self, op: &'static str, a: Val<'py>, b: Val<'py>, node: &Bound<'py, PyAny>) -> PyResult<Val<'py>> {
        let like_b = if let Val::V(v) = b { Some(v) } else { None };
        let a = self.to_value(a, like_b, node)?;
        let b = self.to_value(b, Some(a), node)?;
        let (a, b) = self.broadcast_pair(a, b, node)?;
        let shape = self.func.ty(a).shape.clone();
        Ok(self.v("compare", vec![a, b], Type { elem: Elem::I1, ptr: false, shape }, vec![("kind", Attr::Str(op))]))
    }
}
