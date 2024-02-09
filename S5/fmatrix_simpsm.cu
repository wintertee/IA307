#include "fmatrix.cuh"
#include <assert.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024
__global__ void fmatrix_simpsm_kernel(fmatrix P)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < P.cols)
    {
        float sum = 0.;
        for (int i = 0; i < P.rows; ++i)
        {
            getfm(P, i, j) = expf(getfm(P, i, j));
            sum += getfm(P, i, j)
        }
        for (int i = 0; i < P.rows; ++i)
        {
            getfm(P, i, j) /= sum
        }
    }
}

void fmatrix_simpsm(fmatrix P)
{
    fmatrix_assert(P);
    int threadsPerBlock = P.cols;
    int blocksPerGrid = 1;
    if (threadsPerBlock > THREADS_PER_BLOCK)
    {
        blocksPerGrid = (threadsPerBlock - 1) / THREADS_PER_BLOCK + 1;
        threadsPerBlock = THREADS_PER_BLOCK;
    }
    fmatrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(P);
    gpuErrchk(cudaPeekAtLastError());
}
