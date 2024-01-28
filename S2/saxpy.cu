/*
 * Application saxpy avec GPU
 * y = A.x + y
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#include "cuda_stuff.cuh"

////////////////////////////////////////////////////////////////
//     Initialisation des vecteurs
////////////////////////////////////////////////////////////////
void init_tab(float *tab, int len, float val)
{
    for (int k = 0; k < len; k++)
        tab[k] = val;
}

void print_tab(const char *chaine, float *tab, int len)
{
    int k;
    printf("\nLes 10 premiers de %s: \n", chaine);
    for (k = 0; k < 10; k++)
        printf("%.2f ", tab[k]);
    printf("\nLes 10 derniers: \n");
    for (k = len - 10; k < len; k++)
        printf("%.2f ", tab[k]);
    printf("\n");
}

__global__ void saxpy(float *tabX, float *tabY, int len, float a)
{
    // TODO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len)
        tabY[idx] = tabY[idx] + a * tabX[idx];
}

int main(int argc, char **argv)
{
    float *tabX_d, *tabX_h;
    float *tabY_d, *tabY_h;
    int len = int(pow(10,8));

    /** Initialisation des variables nbthreadbyblock et nbblockbygrid **/
    // TODO
    int block_dim = 128;
    dim3 grid(ceil(len / float(block_dim)));
    dim3 block(block_dim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("SAXPY - tableau de %d éléments \n", len);

    /** Allocation memoire sur le host(CPU) **/
    tabX_h = (float *)malloc(sizeof(float) * len);
    init_tab(tabX_h, len, 5.);
    // TODO - allocation de tabY_h
    tabY_h = (float *)malloc(sizeof(float) * len);
    init_tab(tabY_h, len, 2.);

    /** Affichage initial **/
    printf("Affichage initial\n");
    print_tab("tabX_h", tabX_h, len);
    print_tab("tabY_h", tabY_h, len);

    /** Allocation memoire sur le device(GPU) **/
    gpuErrchk(cudaMalloc((void **)&tabX_d, sizeof(float) * len));
    // TODO - allocation de tabY_d
    gpuErrchk(cudaMalloc((void **)&tabY_d, sizeof(float) * len));

    /** Transfert mémoire du host vers le device **/
    // TODO
    gpuErrchk(cudaMemcpy(tabX_d, tabX_h, sizeof(float) * len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tabY_d, tabY_h, sizeof(float) * len, cudaMemcpyHostToDevice));

    /** Lancement du kernel **/
    // TODO
    cudaEventRecord(start);
    saxpy<<<grid, block>>>(tabX_d, tabY_d, len, 2.);
    cudaEventRecord(stop);

    /** Transfert mémoire du device vers le host **/
    // TODO
    gpuErrchk(cudaMemcpy(tabY_h, tabY_d, sizeof(float) * len, cudaMemcpyDeviceToHost));

    /** Affichage du resultat **/
    printf("Affichage du résultat\n");
    print_tab("tabY_h", tabY_h, len);

    /** Libération de la mémoire **/
    cudaFree(tabX_d);
    cudaFree(tabY_d);
    free(tabX_h);
    free(tabY_h);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Temps d'execution : %f ms\n", milliseconds);
    printf("Fin du programme\n");
    return EXIT_SUCCESS;
}