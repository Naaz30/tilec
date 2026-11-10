mod codegen;
mod frontend;
mod ir;
mod mlir;

use std::collections::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ir::{Elem, Type};

create_exception!(_core, CompileError, PyException, "A kernel could not be compiled.");

fn parse_type(s: &str) -> PyResult<Type> {
    match s {
        "ptr<f32>" => Ok(Type { elem: Elem::F32, ptr: true, shape: vec![] }),
        "i32" => Ok(Type::scalar(Elem::I32)),
        _ => Err(CompileError::new_err(format!("unsupported parameter type {s}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (fdef, param_types, constexprs, globals, source, first_line, num_warps=4))]
fn compile<'py>(fdef: Bound<'py, PyAny>, param_types: HashMap<String, String>,
                constexprs: Bound<'py, PyDict>, globals: Bound<'py, PyDict>,
                source: &str, first_line: usize, num_warps: usize) -> PyResult<(String, String)> {
    let types = param_types.iter()
        .map(|(k, v)| Ok((k.clone(), parse_type(v)?)))
        .collect::<PyResult<HashMap<_, _>>>()?;
    let name: String = fdef.getattr("name")?.extract()?;
    let mut fe = frontend::FrontEnd::new(&name, globals, source, first_line);
    fe.build(&fdef, &types, &constexprs)?;
    let ir = fe.func.to_string();
    if let Ok(tool) = std::env::var("TILEC_OPT") {
        let (opt, msl) = mlir::lower(&tool, &mlir::generate(&fe.func, num_warps)?)?;
        return Ok((format!("{ir}\n\n{opt}"), msl));
    }
    let msl = codegen::generate(&fe.func, num_warps)?;
    Ok((ir, msl))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;
    Ok(())
}
