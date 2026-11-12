#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>

using namespace mlir;

static bool optimize(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  return succeeded(pm.run(module));
}

static llvm::DenseSet<Value> readOnlyBuffers(gpu::GPUFuncOp func) {
  llvm::DenseSet<Value> ro;
  for (BlockArgument arg : func.getArguments().take_front(func.getFunctionType().getNumInputs())) {
    bool written = false;
    for (Operation *user : arg.getUsers())
      if (auto st = dyn_cast<memref::StoreOp>(user))
        written |= st.getMemRef() == arg;
    if (!written)
      ro.insert(arg);
  }
  return ro;
}

static bool hoistable(Operation *op, const llvm::DenseSet<Value> &ro) {
  if (op->getNumRegions() || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (auto ld = dyn_cast<memref::LoadOp>(op))
    return ro.contains(ld.getMemRef());
  return isMemoryEffectFree(op) && isSpeculatable(op);
}

static int hoistLoads(ModuleOp module) {
  int moved = 0;
  module.walk([&](gpu::GPUFuncOp func) {
    llvm::DenseSet<Value> ro = readOnlyBuffers(func);
    func.walk([&](scf::ForOp loop) {
      bool changed = true;
      while (changed) {
        changed = false;
        for (Operation &op : llvm::make_early_inc_range(loop.getBody()->getOperations())) {
          bool outside = llvm::all_of(op.getOperands(), [&](Value v) { return loop.isDefinedOutsideOfLoop(v); });
          if (outside && hoistable(&op, ro)) {
            op.moveBefore(loop);
            changed = true;
            ++moved;
          }
        }
      }
    });
  });
  return moved;
}

struct Emitter {
  llvm::DenseMap<Value, std::string> names;
  std::string out;
  std::string err;
  int counter = 0;
  int indent = 1;

  void line(const std::string &s) { out += std::string(indent * 4, ' ') + s + "\n"; }

  std::string fresh() { return "v" + std::to_string(counter++); }

  std::string type(Type t) {
    if (t.isF32()) return "float";
    if (t.isInteger(1)) return "bool";
    if (t.isInteger(32) || t.isIndex()) return "int";
    std::string s;
    llvm::raw_string_ostream os(s);
    t.print(os);
    err = "unsupported type " + s;
    return "?";
  }

  std::string name(Value v) { return names.lookup(v); }

  void define(Value v, const std::string &expr) {
    std::string n = fresh();
    names[v] = n;
    line("const " + type(v.getType()) + " " + n + " = " + expr + ";");
  }

  std::string literal(Attribute a, Type t) {
    if (auto i = dyn_cast<IntegerAttr>(a)) {
      if (t.isInteger(1)) return i.getValue().isZero() ? "false" : "true";
      return std::to_string(i.getValue().getSExtValue());
    }
    if (auto f = dyn_cast<FloatAttr>(a)) {
      double d = f.getValueAsDouble();
      if (std::isnan(d)) return "NAN";
      if (std::isinf(d)) return d > 0 ? "INFINITY" : "-INFINITY";
      char buf[64];
      std::snprintf(buf, sizeof buf, "float(%.9g)", d);
      return buf;
    }
    err = "unsupported constant";
    return "?";
  }

  bool emitBlock(Block &block) {
    for (Operation &op : block) {
      if (!emitOp(&op)) return false;
      if (!err.empty()) return false;
    }
    return true;
  }

  bool emitOp(Operation *op) {
    static const std::map<std::string, std::string> binary = {
        {"arith.addi", "+"}, {"arith.subi", "-"}, {"arith.muli", "*"}, {"arith.divsi", "/"},
        {"arith.remsi", "%"}, {"arith.andi", "&"}, {"arith.ori", "|"}, {"arith.xori", "^"},
        {"arith.shli", "<<"}, {"arith.shrsi", ">>"}, {"arith.addf", "+"}, {"arith.subf", "-"},
        {"arith.mulf", "*"}, {"arith.divf", "/"}};
    static const std::map<std::string, std::string> unsignedOps = {
        {"arith.divui", "/"}, {"arith.remui", "%"}, {"arith.shrui", ">>"}};
    static const std::map<std::string, std::string> minmax = {
        {"arith.maxsi", "max"}, {"arith.minsi", "min"}, {"arith.maximumf", "max"},
        {"arith.minimumf", "min"}, {"arith.maxnumf", "max"}, {"arith.minnumf", "min"}};
    static const std::map<std::string, std::string> icmp = {
        {"eq", "=="}, {"ne", "!="}, {"slt", "<"}, {"sle", "<="}, {"sgt", ">"}, {"sge", ">="},
        {"ult", "<"}, {"ule", "<="}, {"ugt", ">"}, {"uge", ">="}};
    static const std::map<std::string, std::string> fcmp = {
        {"oeq", "=="}, {"one", "!="}, {"olt", "<"}, {"ole", "<="}, {"ogt", ">"}, {"oge", ">="},
        {"ueq", "=="}, {"une", "!="}, {"ult", "<"}, {"ule", "<="}, {"ugt", ">"}, {"uge", ">="}};
    static const std::map<std::string, std::string> reduce = {
        {"add", "simd_sum"}, {"mul", "simd_product"}, {"minsi", "simd_min"}, {"maxsi", "simd_max"},
        {"minimumf", "simd_min"}, {"maximumf", "simd_max"}, {"minnumf", "simd_min"}, {"maxnumf", "simd_max"}};

    std::string opname = op->getName().getStringRef().str();
    auto operand = [&](unsigned i) { return name(op->getOperand(i)); };

    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      define(c.getResult(), literal(c.getValue(), c.getType()));
    } else if (binary.count(opname)) {
      define(op->getResult(0), "(" + operand(0) + " " + binary.at(opname) + " " + operand(1) + ")");
    } else if (unsignedOps.count(opname)) {
      define(op->getResult(0), "int(uint(" + operand(0) + ") " + unsignedOps.at(opname) + " uint(" + operand(1) + "))");
    } else if (minmax.count(opname)) {
      define(op->getResult(0), minmax.at(opname) + "(" + operand(0) + ", " + operand(1) + ")");
    } else if (opname == "arith.negf") {
      define(op->getResult(0), "(-" + operand(0) + ")");
    } else if (auto c = dyn_cast<arith::CmpIOp>(op)) {
      std::string p = arith::stringifyCmpIPredicate(c.getPredicate()).str();
      std::string a = operand(0), b = operand(1);
      if (p[0] == 'u') a = "uint(" + a + ")", b = "uint(" + b + ")";
      define(c.getResult(), "(" + a + " " + icmp.at(p) + " " + b + ")");
    } else if (auto c = dyn_cast<arith::CmpFOp>(op)) {
      std::string p = arith::stringifyCmpFPredicate(c.getPredicate()).str();
      if (!fcmp.count(p)) { err = "unsupported cmpf " + p; return false; }
      define(c.getResult(), "(" + operand(0) + " " + fcmp.at(p) + " " + operand(1) + ")");
    } else if (isa<arith::SelectOp>(op)) {
      define(op->getResult(0), "(" + operand(0) + " ? " + operand(1) + " : " + operand(2) + ")");
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                   arith::SIToFPOp, arith::FPToSIOp>(op)) {
      define(op->getResult(0), type(op->getResult(0).getType()) + "(" + operand(0) + ")");
    } else if (auto t = dyn_cast<gpu::ThreadIdOp>(op)) {
      if (t.getDimension() != gpu::Dimension::x) { err = "only gpu.thread_id x"; return false; }
      define(t.getResult(), "int(tid3.x)");
    } else if (auto b = dyn_cast<gpu::BlockIdOp>(op)) {
      std::string d = b.getDimension() == gpu::Dimension::x ? "x" : b.getDimension() == gpu::Dimension::y ? "y" : "z";
      define(b.getResult(), "int(tgpig." + d + ")");
    } else if (auto r = dyn_cast<gpu::SubgroupReduceOp>(op)) {
      std::string k = gpu::stringifyAllReduceOperation(r.getOp()).str();
      if (!reduce.count(k)) { err = "unsupported subgroup_reduce " + k; return false; }
      define(r.getResult(), reduce.at(k) + "(" + operand(0) + ")");
    } else if (isa<gpu::BarrierOp>(op)) {
      line("threadgroup_barrier(mem_flags::mem_threadgroup);");
    } else if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      define(ld.getResult(), name(ld.getMemRef()) + "[" + name(ld.getIndices()[0]) + "]");
    } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
      line(name(st.getMemRef()) + "[" + name(st.getIndices()[0]) + "] = " + name(st.getValue()) + ";");
    } else if (auto f = dyn_cast<scf::ForOp>(op)) {
      for (auto [arg, init, res] : llvm::zip(f.getRegionIterArgs(), f.getInitArgs(), f.getResults())) {
        std::string n = fresh();
        line(type(arg.getType()) + " " + n + " = " + name(init) + ";");
        names[arg] = n;
        names[res] = n;
      }
      std::string iv = fresh();
      names[f.getInductionVar()] = iv;
      line("for (int " + iv + " = " + name(f.getLowerBound()) + "; " + iv + " < " + name(f.getUpperBound()) +
           "; " + iv + " += " + name(f.getStep()) + ") {");
      ++indent;
      if (!emitBlock(*f.getBody())) return false;
      --indent;
      line("}");
    } else if (auto i = dyn_cast<scf::IfOp>(op)) {
      std::vector<std::string> results;
      for (Value r : i.getResults()) {
        std::string n = fresh();
        names[r] = n;
        results.push_back(n);
        line(type(r.getType()) + " " + n + ";");
      }
      line("if (" + name(i.getCondition()) + ") {");
      ++indent;
      if (!emitBlock(i.getThenRegion().front())) return false;
      --indent;
      if (!i.getElseRegion().empty()) {
        line("} else {");
        ++indent;
        if (!emitBlock(i.getElseRegion().front())) return false;
        --indent;
      }
      line("}");
    } else if (auto y = dyn_cast<scf::YieldOp>(op)) {
      if (auto parent = dyn_cast<scf::IfOp>(y->getParentOp())) {
        for (unsigned k = 0; k < y.getNumOperands(); ++k)
          line(name(parent.getResult(k)) + " = " + operand(k) + ";");
      } else if (auto loop = dyn_cast<scf::ForOp>(y->getParentOp())) {
        std::vector<std::string> temps;
        for (unsigned k = 0; k < y.getNumOperands(); ++k) {
          std::string t = fresh();
          line("const " + type(y.getOperand(k).getType()) + " " + t + " = " + operand(k) + ";");
          temps.push_back(t);
        }
        for (unsigned k = 0; k < y.getNumOperands(); ++k)
          line(name(loop.getRegionIterArgs()[k]) + " = " + temps[k] + ";");
      }
    } else if (isa<gpu::ReturnOp>(op)) {
    } else {
      err = "Metal emitter: unsupported op " + opname;
      return false;
    }
    return true;
  }

  bool emitFunc(gpu::GPUFuncOp func) {
    llvm::DenseSet<Value> ro = readOnlyBuffers(func);
    std::vector<std::string> argNames;
    if (auto a = func->getAttrOfType<ArrayAttr>("tilec.arg_names"))
      for (Attribute n : a) argNames.push_back(cast<StringAttr>(n).str());
    std::vector<std::string> params;
    for (auto [i, arg] : llvm::enumerate(func.getArguments().take_front(func.getFunctionType().getNumInputs()))) {
      auto mt = cast<MemRefType>(arg.getType());
      std::string n = i < argNames.size() ? argNames[i] : "arg" + std::to_string(i);
      std::string el = type(mt.getElementType());
      bool scalar = mt.hasStaticShape() && mt.getNumElements() == 1;
      if (scalar) n += "_arg";
      names[arg] = n;
      params.push_back(std::string("device ") + (ro.contains(arg) ? "const " : "") + el + "* " + n + " [[buffer(" +
                       std::to_string(i) + ")]]");
    }
    params.push_back("uint3 tgpig [[threadgroup_position_in_grid]]");
    params.push_back("uint3 tid3 [[thread_position_in_threadgroup]]");
    params.push_back("uint lane [[thread_index_in_simdgroup]]");
    params.push_back("uint sg [[simdgroup_index_in_threadgroup]]");

    std::string info;
    if (auto s = func->getAttrOfType<StringAttr>("tilec.info")) info = s.str();
    out += "#include <metal_stdlib>\nusing namespace metal;\n\n";
    out += "// generated by tilec (MLIR)  |  " + info + "\n";
    out += "kernel void " + func.getName().str() + "(\n";
    for (size_t k = 0; k < params.size(); ++k)
      out += "    " + params[k] + (k + 1 < params.size() ? ",\n" : ")\n");
    out += "{\n";
    for (auto [k, wg] : llvm::enumerate(func.getWorkgroupAttributions())) {
      auto mt = cast<MemRefType>(wg.getType());
      std::string n = "red" + std::to_string(k);
      names[wg] = n;
      line("threadgroup " + type(mt.getElementType()) + " " + n + "[" + std::to_string(mt.getNumElements()) + "];");
    }
    if (!emitBlock(func.getBody().front())) return false;
    out += "}\n";
    return err.empty();
  }
};

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto input = llvm::MemoryBuffer::getSTDIN();
  if (!input) {
    llvm::errs() << "cannot read stdin\n";
    return 1;
  }
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*input), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &ctx);
  if (!module) return 1;

  bool noHoist = argc > 1 && std::string(argv[1]) == "--no-hoist";
  if (!optimize(*module)) return 1;
  int moved = noHoist ? 0 : hoistLoads(*module);
  if (moved && !optimize(*module)) return 1;

  module->print(llvm::outs());
  llvm::outs() << "\n// hoisted " << moved << " ops out of loops\n//--MSL--\n";

  Emitter em;
  bool ok = true;
  module->walk([&](gpu::GPUFuncOp f) { ok = ok && em.emitFunc(f); });
  if (!ok) {
    llvm::errs() << em.err << "\n";
    return 1;
  }
  llvm::outs() << em.out;
  return 0;
}
