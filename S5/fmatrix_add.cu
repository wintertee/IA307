#include "fmatrix.cuh"
#include <assert.h>

#define THREADS_PER_BLOCK 1024
__global__
void fmatrix_add_kernel(fmatrix P,float a,fmatrix Y) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int j = idx / P.rows;
    int i = idx % P.rows;
    if (i < P.rows && j < P.cols ){
        getfm(P,i,j) += a*getfm(Y,i,j);
    }
}

/** Compute P = P + a*Y */
void fmatrix_add(fmatrix P,float a,fmatrix Y) {
    fmatrix_assert(P);
    fmatrix_assert(Y);
    assert(P.rows == Y.rows);
    assert(P.cols == Y.cols);
    int threadsPerBlock = fmatrix_elements(P);
    int blocksPerGrid = 1;
    if (threadsPerBlock > THREADS_PER_BLOCK){
        blocksPerGrid = (threadsPerBlock-1)/THREADS_PER_BLOCK+1;
        threadsPerBlock = THREADS_PER_BLOCK;
    }
    fmatrix_add_kernel<<< blocksPerGrid, threadsPerBlock >>>(P,a,Y);
    gpuErrchk( cudaPeekAtLastError() );
}
