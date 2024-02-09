#include "fmatrix.cuh"
#include "fmatrix_mult.cu"

int main() {
    // read two matrices on the host from csv files
    fmatrix B = fmatrix_device_from_csv("test_B.csv");
    fmatrix C = fmatrix_device_from_csv("test_C.csv");
    // allocate matrix A
    fmatrix A = fmatrix_create_on_device(B.rows,C.cols);

    // compute A = 3.0*B*C
    fmatrix_mult(A,3.0,B,C);

    fmatrix_device_to_csv("test_A.csv",A);

    fmatrix_free_on_device(&B);
    fmatrix_free_on_device(&C);
    fmatrix_free_on_device(&A);
}
