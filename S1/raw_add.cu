#include <stdio.h>
#include <stdlib.h>

__global__ void add(int a, int b, int *res)
{
    *res = a + b;
}

int main()
{
    int res = 0;
    int *d_res = NULL;

    // Launch add() kernel on GPU
    add<<<1, 1>>>(2, 2, d_res);

    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 + 2 = %d\n", res);

    return EXIT_SUCCESS;
}