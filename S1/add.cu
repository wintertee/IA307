#include <stdio.h>
#include <stdlib.h>

__global__ void add(int a, int b, int *res)
{
    *res = a + b;
}

int main()
{
    printf("hello world\n");
    int res = 0;
    int *d_res = NULL;
    cudaError_t err;

    // Launch add() kernel on GPU
    add<<<1, 1>>>(2, 2, d_res);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stdout, "GPUassert: add launch failed with the error : %s \n", cudaGetErrorString(err));
        exit(err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stdout, "GPUassert: add execution failed with the error : %s \n", cudaGetErrorString(err));
        exit(err);
    }

    err = cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stdout, "GPUassert: cudaMemcpy failed with the error : %s \n", cudaGetErrorString(err));
        exit(err);
    }

    printf("2 + 2 = %d\n", res);

    return EXIT_SUCCESS;
}