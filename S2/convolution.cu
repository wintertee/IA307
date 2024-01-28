#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

static int N = 100;
#define NB_STEPS 50

double **cur_step;
double **next_step;

double weight[3][3] = {{0, 2, 0}, {2, 4, 2}, {0, 2, 0}};

/* initialize matrixes */
void init()
{
    int i, j;

    cur_step = (double **)malloc(sizeof(double *) * N);
    next_step = (double **)malloc(sizeof(double *) * N);

    srand(0);
    for (i = 0; i < N; i++)
    {
        cur_step[i] = (double *)calloc(N, sizeof(double));
        next_step[i] = (double *)calloc(N, sizeof(double));
    }

    int nb_hot_spots = rand() % N;
    printf("Generating %d hot spots\n", nb_hot_spots);
    for (i = 0; i < nb_hot_spots; i++)
    {
        int posx = rand() % N;
        int posy = rand() % N;
        cur_step[posx][posy] += N * 10000;
        printf("%d,%d : %lf\n", posx, posy, cur_step[posx][posy]);
    }
}

/* dump the matrix in f */
void print_matrix(FILE *f, double **matrix)
{
    int i, j;
    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            fprintf(f, " %.2f  ", matrix[i][j]);
        }
        fprintf(f, "\n");
    }
}

void compute()
{
    int i, j;

    for (i = 1; i < N - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            next_step[i][j] = (cur_step[i - 1][j] * weight[0][1] + cur_step[i + 1][j] * weight[2][1] + cur_step[i][j - 1] * weight[1][0] + cur_step[i][j + 1] * weight[1][2] + cur_step[i][j] * weight[1][1]) / 5;
        }
    }

    /* swap buffers */
    double **tmp = cur_step;
    cur_step = next_step;
    next_step = tmp;
}


int main(int argc, char **argv)
{

    char *output_file = "result.dat";
    int dump = 1;

    struct timeval t1, t2;
    init();

    gettimeofday(&t1, NULL);
    for (int i = 0; i < NB_STEPS; i++)
    {
        printf("STEP %d...\n", i);
        compute();
    }
    gettimeofday(&t2, NULL);

    double total_time = ((t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec)) / 1e6;
    printf("%d steps in %lf sec (%lf sec/step)\n", NB_STEPS, total_time, total_time / NB_STEPS);

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