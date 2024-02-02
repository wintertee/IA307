#ifndef sgemm_H
#define sgemm_H

#include <string>

#include "cublas_v2.h"
#include "fmatrix.cuh"

void mat_mul(cublasHandle_t handle, fmatrix A, fmatrix B, fmatrix C, std::string arg);

#endif