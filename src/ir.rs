use std::fmt;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    I1,  
    I32,
    F32, 
}

impl Elem {
    pub fn name(self) -> &'static str {
        match self {
            Elem::I1 => "i1",
            Elem::I32 => "i32",
            Elem::F32 => "f32",
        }
    }
    pub fn msl(self) -> &'static str {
        match self {
            Elem::I1 => "bool",
            Elem::I32 => "int",
            Elem::F32 => "float",
        }
    }
    pub fn is_int(self) -> bool {
        self == Elem::I32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Type {
    pub elem: Elem,
    pub ptr: bool,
    pub shape: Vec<usize>,
}

impl Type {
    pub fn scalar(elem: Elem) -> Type {
        Type { elem, ptr: false, shape: vec![] }
    }
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn with_shape(&self, shape: &[usize]) -> Type {
        Type { shape: shape.to_vec(), ..self.clone() }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let elem = if self.ptr { format!("ptr<{}>", self.elem.name()) } else { self.elem.name().to_string() };
        if self.is_scalar() {
            return write!(f, "{elem}");
        }
        let dims: Vec<String> = self.shape.iter().map(|s| s.to_string()).collect();
        write!(f, "tensor<{}x{elem}>", dims.join("x"))
    }
}


#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Str(&'static str),
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Attr::Int(i) => write!(f, "{i}"),
            Attr::Float(x) => write!(f, "{x:?}"),
            Attr::Str(s) => write!(f, "'{s}'"),
        }
    }
}


#[derive(Clone, Debug)]
pub struct Op {
    pub opcode: &'static str,
    pub operands: Vec<usize>,
    pub attrs: Vec<(&'static str, Attr)>,
    pub result: Option<usize>,
}

impl Op {
    pub fn attr(&self, name: &str) -> &Attr {
        &self.attrs.iter().find(|(k, _)| *k == name).expect("missing attribute").1
    }
}


pub struct Function {
    pub name: String,
    pub params: Vec<(String, usize)>,
    pub constexprs: Vec<(String, String)>,
    pub ops: Vec<Op>,
    pub types: Vec<Type>,
}

impl Function {
    pub fn new(name: &str) -> Function {
        Function { name: name.to_string(), params: vec![], constexprs: vec![], ops: vec![], types: vec![] }
    }

    fn new_value(&mut self, ty: Type) -> usize {
        self.types.push(ty);
        self.types.len() - 1
    }

    pub fn add_param(&mut self, name: &str, ty: Type) -> usize {
        let v = self.new_value(ty);
        self.params.push((name.to_string(), v));
        v
    }

    pub fn emit(&mut self, opcode: &'static str, operands: Vec<usize>, ty: Option<Type>,
                attrs: Vec<(&'static str, Attr)>) -> Option<usize> {
        let result = ty.map(|t| self.new_value(t));
        self.ops.push(Op { opcode, operands, attrs, result });
        result
    }

    pub fn ty(&self, v: usize) -> &Type {
        &self.types[v]
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let params: Vec<String> = self.params.iter()
            .map(|(n, v)| format!("%{v}: {}  /* {n} */", self.ty(*v)))
            .collect();
        write!(f, "func @{}({})", self.name, params.join(", "))?;
        if !self.constexprs.is_empty() {
            let cs: Vec<String> = self.constexprs.iter().map(|(k, v)| format!("{k}={v}")).collect();
            write!(f, "\n  // constexpr: {}", cs.join(", "))?;
        }
        for op in &self.ops {
            let args: Vec<String> = op.operands.iter().map(|o| format!("%{o}")).collect();
            let attrs = if op.attrs.is_empty() {
                String::new()
            } else {
                let a: Vec<String> = op.attrs.iter().map(|(k, v)| format!("{k}={v}")).collect();
                format!(" {{{}}}", a.join(", "))
            };
            match op.result {
                Some(r) => write!(f, "\n  %{r} = {}({}){attrs} : {}", op.opcode, args.join(", "), self.ty(r))?,
                None => write!(f, "\n  {}({}){attrs}", op.opcode, args.join(", "))?,
            }
        }
        Ok(())
    }
}
