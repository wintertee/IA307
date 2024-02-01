#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_stuff.cuh"

static int N = 100;
#define NB_STEPS 50

double *cur_step;
double *next_step;

double *cur_step_d, *next_step_d;

double weight[9] = {0, 2, 0, 2, 4, 2, 0, 2, 0};
__device__ __constant__ double weight_d[9];

/* initialize matrixes */
void init()
{
    int i;

    cur_step = (double *)calloc(N * N, sizeof(double));
    next_step = (double *)calloc(N * N, sizeof(double));

    srand(0);

    int nb_hot_spots = rand() % N;
    printf("Generating %d hot spots\n", nb_hot_spots);
    for (i = 0; i < nb_hot_spots; i++)
    {
        int posx = rand() % N;
        int posy = rand() % N;
        cur_step[posx * N + posy] += N * 10000;
        printf("%d,%d : %lf\n", posx, posy, cur_step[posx * N + posy]);
    }
}

/* dump the matrix in f */
void print_matrix(FILE *f, double *matrix)
{
    int i, j;
    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            fprintf(f, " %.2f  ", matrix[i * N + j]);
        }
        fprintf(f, "\n");
    }
}

__global__ void compute(double *cur_step_d, double *next_step_d, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= 1 && i < N - 1)
    {
        if (j >= 1 && j < N - 1)
        {
            next_step_d[i * N + j] = cur_step_d[(i - 1) * N + j] * weight_d[0 * 3 + 1];
            next_step_d[i * N + j] += cur_step_d[(i + 1) * N + j] * weight_d[2 * 3 + 1];
            next_step_d[i * N + j] += cur_step_d[i * N + j - 1] * weight_d[1 * 3 + 0];
            next_step_d[i * N + j] += cur_step_d[i * N + j + 1] * weight_d[1 * 3 + 2];
            next_step_d[i * N + j] += cur_step_d[i * N + j] * weight_d[1 * 3 + 1];
            next_step_d[i * N + j] /= 5;
        }
    }
}

int main(int argc, char **argv)
{

    const char *output_file = "resultgpu.dat";
    int dump = 1;

    int block_dim = 32;
    dim3 block(block_dim, block_dim);
    dim3 grid(ceil(N / float(block_dim)), ceil(N / float(block_dim)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    init();

    gpuErrchk(cudaMalloc((void **)&cur_step_d, N * N * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&next_step_d, N * N * sizeof(double)));

    gpuErrchk(cudaMemcpyToSymbol(weight_d, weight, 9 * sizeof(double)));
    gpuErrchk(cudaMemcpy(cur_step_d, cur_step, N * N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(next_step_d, next_step, N * N * sizeof(double), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    for (int i = 0; i < NB_STEPS; i++)
    {
        printf("STEP %d...\n", i);
        compute<<<grid, block>>>(cur_step_d, next_step_d, N);
        /* swap buffers */
        double *tmp = cur_step_d;
        cur_step_d = next_step_d;
        next_step_d = tmp;
    }
    cudaEventRecord(stop);

    gpuErrchk(cudaMemcpy(cur_step, cur_step_d, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(next_step, next_step_d, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(cur_step_d));
    gpuErrchk(cudaFree(next_step_d));
    cudaEventSynchronize(stop);
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    printf("%d steps in %lf sec (%lf sec/step)\n", NB_STEPS, total_time / 1000, total_time / 1000 / NB_STEPS);

    if (dump)
    {
        printf("dumping the result data in %s\n", output_file);
        FILE *f = fopen(output_file, "w");
        print_matrix(f, cur_step);
    }
    else
    {
        print_matrix(stdout, cur_step);
    }

    return EXIT_SUCCESS;
}