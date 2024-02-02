#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cuda_stuff.cuh"
#include "sgemm.cuh"
#include "fmatrix.cuh"

#define THREADS_PER_BLOCK 1024
#define TILE_WIDTH 32

#define CEILING(x, y) (((x) + (y)-1) / (y))

using namespace std;


/* basic matrix multiplication C = alpha*A*B + beta*C on host as reference for the speedup */
void matrixMultiplication_basic_host(float alpha, fmatrix A, fmatrix B, float beta, fmatrix C)
{
    float tmp = 0;
    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < B.cols; j++)
        {
            for (int k = 0; k < A.cols; k++)
            {
                tmp += alpha * getfm(A, i, k) * getfm(B, k, j);
            }
            getfm(C, i, j) = beta * getfm(C, i, j) + tmp;
            tmp = 0;
        }
    }
}

/* TODO : 3 different versions of matrix multiplication C = alpha*A*B + beta*C on device */
__global__ void matmul_basic_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB)
{
    /* TODO */
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nb_ColB && y < nb_LigneA)
    {
        float tmp = 0;
        for (int k = 0; k < nb_ColA; k++)
        {
            tmp += A[k * nb_ColA + x] * B[y * nb_ColB + k];
        }
        C[y * nb_ColB + x] = alpha * tmp + beta * C[y * nb_ColB + x];
    }
}

void matrixMultiplication_basic(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C)
{
    // TODO - declaration of dimGrid and dimBlock
    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil(d_B.cols / 32), ceil(d_A.rows / 32));

    matmul_basic_kernel<<<dimGrid, dimBlock>>>(alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
}

/**********************/
__global__ void matmul_tiled_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB)
{
    /* TODO */
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int nb_tile_in_block = CEILING(nb_ColA, TILE_WIDTH);
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0;

    for (int i = 0; i < nb_tile_in_block; i++)
    {
        // Cooperative data loading in the SAME shared tile 
        if (y < nb_LigneA && i * TILE_WIDTH + threadIdx.x < nb_ColA)
        {
            s_A[threadIdx.y][threadIdx.x] = A[y * nb_ColA + i * TILE_WIDTH + threadIdx.x];
        }
        else
        {
            s_A[threadIdx.y][threadIdx.x] = 0;
        }
        if (x < nb_ColB && i * TILE_WIDTH + threadIdx.y < nb_LigneB)
        {
            s_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * nb_ColB + x];
        }
        else
        {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        // Accumulation of the partial result in local variable tmp
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            tmp += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (x < nb_ColB && y < nb_LigneA)
    {
        C[y * nb_ColB + x] = alpha * tmp + beta * C[y * nb_ColB + x];
    }
}

void matrixMultiplication_tiled(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C)
{
    // TODO - declaration of dimGrid and dimBlock
    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil(d_B.cols / 32), ceil(d_A.rows / 32));

    matmul_tiled_kernel<<<dimGrid, dimBlock>>>(alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
}

/**********************/
void matrixMultiplication_cublas(cublasHandle_t handle, float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C)
{
    /* TODO */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d_A.rows, d_B.cols, d_A.cols, &alpha, d_A.data, d_A.rows, d_B.data, d_B.rows, &beta, d_C.data, d_C.rows);
}

/*MAIN SGEMM*/
void gen_mat_mul(cublasHandle_t handle, float alpha, fmatrix A, fmatrix B, float beta, fmatrix C, std::string arg)
{
    if (arg == "cpu")
    {
        matrixMultiplication_basic_host(alpha, A, B, beta, C);
    }
    else
    {
        /* kernel function*/
        if (arg == "gpu_basic")
        {
            matrixMultiplication_basic(alpha, A, B, beta, C);
        }
        else if (arg == "gpu_tiled")
        {
            matrixMultiplication_tiled(alpha, A, B, beta, C);
        }
        else if (arg == "gpu_cublas")
        {
            matrixMultiplication_cublas(handle, alpha, A, B, beta, C);
        }
        else
        {
            printf("Matrix Multiplication argument is Wrong");
            exit(0);
        }
        // wait for everything to finish
        device_synchronize();
    }
}

void mat_mul(cublasHandle_t handle, fmatrix A, fmatrix B, fmatrix C, std::string arg)
{
    gen_mat_mul(handle, 1.0, A, B, 0.0, C, arg);
}
