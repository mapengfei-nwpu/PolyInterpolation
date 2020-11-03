#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

__constant__ double cudaG_inv[100];

__global__ void outputG_inv_device(
){
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if(index == 0){
        printf("G_inv:\n");
        for(size_t i = 0; i < 10; i++){
            for(size_t j = 0; j < 10; j++){
                printf(" %f ",cudaG_inv[i*10+j]);
            }
            printf("\n");
        }
    }
}

__device__ void linear_transformation(double *p, double *F, double *b)
{

    // transpose p
    double temp = p[2];
    p[2] = p[6];
    p[6] = p[7];
    p[7] = p[10];
    p[10] = p[8];
    p[8] = temp;
    temp = p[1];
    p[1] = p[3];
    p[3] = p[9];
    p[9] = p[5];
    p[5] = p[4];
    p[4] = temp;

    ///**************************************************
    ///        point1    point2    point3    point4
    ///
    /// x  =    x[0]      x[1]      x[2]      x[3]
    /// y  =    y[0]      y[1]      y[2]      y[3]
    /// z  =    z[0]      z[1]      z[2]      z[3]
    ///
    ///**************************************************

    double *x = p;
    double *y = &(p[4]);
    double *z = &(p[8]);

    double det =
        x[0] * y[1] * z[2] - x[0] * y[2] * z[1] - x[1] * y[0] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1] - x[2] * y[1] * z[0] - x[0] * y[1] * z[3] + x[0] * y[3] * z[1] + x[1] * y[0] * z[3] - x[1] * y[3] * z[0] - x[3] * y[0] * z[1] + x[3] * y[1] * z[0] + x[0] * y[2] * z[3] - x[0] * y[3] * z[2] - x[2] * y[0] * z[3] + x[2] * y[3] * z[0] + x[3] * y[0] * z[2] - x[3] * y[2] * z[0] - x[1] * y[2] * z[3] + x[1] * y[3] * z[2] + x[2] * y[1] * z[3] - x[2] * y[3] * z[1] - x[3] * y[1] * z[2] + x[3] * y[2] * z[1];

    F[0 * 3 + 0] = -(y[0] * z[2] - y[2] * z[0] - y[0] * z[3] + y[3] * z[0] + y[2] * z[3] - y[3] * z[2]) / det;
    F[0 * 3 + 1] = (x[0] * z[2] - x[2] * z[0] - x[0] * z[3] + x[3] * z[0] + x[2] * z[3] - x[3] * z[2]) / det;
    F[0 * 3 + 2] = -(x[0] * y[2] - x[2] * y[0] - x[0] * y[3] + x[3] * y[0] + x[2] * y[3] - x[3] * y[2]) / det;

    F[1 * 3 + 0] = (y[0] * z[1] - y[1] * z[0] - y[0] * z[3] + y[3] * z[0] + y[1] * z[3] - y[3] * z[1]) / det;
    F[1 * 3 + 1] = -(x[0] * z[1] - x[1] * z[0] - x[0] * z[3] + x[3] * z[0] + x[1] * z[3] - x[3] * z[1]) / det;
    F[1 * 3 + 2] = (x[0] * y[1] - x[1] * y[0] - x[0] * y[3] + x[3] * y[0] + x[1] * y[3] - x[3] * y[1]) / det;

    F[2 * 3 + 0] = -(y[0] * z[1] - y[1] * z[0] - y[0] * z[2] + y[2] * z[0] + y[1] * z[2] - y[2] * z[1]) / det;
    F[2 * 3 + 1] = (x[0] * z[1] - x[1] * z[0] - x[0] * z[2] + x[2] * z[0] + x[1] * z[2] - x[2] * z[1]) / det;
    F[2 * 3 + 2] = -(x[0] * y[1] - x[1] * y[0] - x[0] * y[2] + x[2] * y[0] + x[1] * y[2] - x[2] * y[1]) / det;

    b[0] = -F[0 * 3 + 0] * x[0] - F[0 * 3 + 1] * y[0] - F[0 * 3 + 2] * z[0];
    b[1] = -F[1 * 3 + 0] * x[0] - F[1 * 3 + 1] * y[0] - F[1 * 3 + 2] * z[0];
    b[2] = -F[2 * 3 + 0] * x[0] - F[2 * 3 + 1] * y[0] - F[2 * 3 + 2] * z[0];
}

__device__ void point_local_to_ref(double *point_out, double *point_in, double *F, double *b)
{

    point_out[0] = F[0 * 3 + 0] * point_in[0] + F[0 * 3 + 1] * point_in[1] + F[0 * 3 + 2] * point_in[2];
    point_out[1] = F[1 * 3 + 0] * point_in[0] + F[1 * 3 + 1] * point_in[1] + F[1 * 3 + 2] * point_in[2];
    point_out[2] = F[2 * 3 + 0] * point_in[0] + F[2 * 3 + 1] * point_in[1] + F[2 * 3 + 2] * point_in[2];

    point_out[0] += b[0];
    point_out[1] += b[1];
    point_out[2] += b[2];
}

__device__ void transform_points(double *points_ref, double *points, double *coordinates_dof, size_t num_points)
{
    double F[9] = {0};
    double b[3] = {0};
    linear_transformation(coordinates_dof, F, b);
    for (size_t i = 0; i < num_points; i++)
    {
        point_local_to_ref(&(points_ref[3 * i]), &(points[3 * i]), F, b);
    }
}

__global__ void transform_points_all_device(
    size_t num_cells,
    size_t num_gauss,
    double *coordinates,
    double *gauss_points,
    double *gauss_points_ref
){
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if(index < num_cells){
        double *points_ref = &(gauss_points_ref[num_gauss * 3 * index]);
        double *points = &(gauss_points[num_gauss * 3 * index]);
        double *cell_coordinates = &(coordinates[12 * index]);
        transform_points(points_ref, points, cell_coordinates, num_gauss);
    }
}

extern "C"{

    void setCudaG_inv(double *G_inv)
    {
        checkCudaErrors(cudaMemcpyToSymbol(cudaG_inv, G_inv, sizeof(double) * 100));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void outputG_inv()
    {
        uint numThreads, numBlocks;
        computeGridSize(1000, 256, numBlocks, numThreads);

        // execute the kernel
        outputG_inv_device<<< numBlocks, numThreads >>>();

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void transform_points_all(
        size_t num_cells,
        size_t num_gauss,
        double *coordinates_host,
        double *gauss_points_host,
        double *gauss_points_ref_host
    ){
        uint numThreads, numBlocks;
        computeGridSize(num_cells, 256, numBlocks, numThreads);

        double *coordinates;
        double *gauss_points;
        double *gauss_points_ref;

        checkCudaErrors(cudaMalloc((void **)&coordinates, 12*num_cells*sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&gauss_points, 3*num_gauss*num_cells*sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&gauss_points_ref, 3*num_gauss*num_cells*sizeof(double)));
        checkCudaErrors(cudaMemcpy(coordinates, coordinates_host, 12*num_cells*sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gauss_points, gauss_points_host, 3*num_gauss*num_cells*sizeof(double), cudaMemcpyHostToDevice));

        transform_points_all_device<<< numBlocks, numThreads >>>(
            num_cells,
            num_gauss,
            coordinates,
            gauss_points,
            gauss_points_ref);

        getLastCudaError("Kernel execution failed");

        checkCudaErrors(cudaMemcpy(gauss_points_ref_host, gauss_points_ref, 3*num_gauss*num_cells*sizeof(double), cudaMemcpyDeviceToHost));
        
        cudaFree(gauss_points);
        cudaFree(gauss_points_ref);
        cudaFree(coordinates);
    }
}


