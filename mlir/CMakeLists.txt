cmake_minimum_required(VERSION 3.20)
project(tilec_opt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR from ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
separate_arguments(LLVM_DEFS NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFS})

add_executable(tilec-opt tilec_opt.cpp)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(tilec-opt PRIVATE -fno-rtti)
endif()
target_link_libraries(tilec-opt PRIVATE
  MLIRParser MLIRPass MLIRTransforms
  MLIRArithDialect MLIRGPUDialect MLIRMemRefDialect MLIRSCFDialect
  MLIRIR MLIRSupport)
