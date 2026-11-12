#pragma once

// clangd-only shim: map common device math calls to clang builtins.
//
// The real CUDA math headers are not parsed (see cuda_runtime.h), and adding
// __device__ declarations of <cmath> functions causes false diagnostics when a
// .cuh header is parsed standalone. Including <cmath> first and then defining
// macros avoids both problems for clangd; nvcc never sees this file.

#include <cmath>

#include <cuda_runtime.h>

#define tanhf __builtin_tanhf
#define expf __builtin_expf
#define logf __builtin_logf
#define sqrtf __builtin_sqrtf
#define fabsf __builtin_fabsf
#define fmaxf __builtin_fmaxf
#define fminf __builtin_fminf
#define powf __builtin_powf
#define erff __builtin_erff
#define fmaf __builtin_fmaf
