#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

__constant__ double cudaG_inv[100];

__global__ void outputG_inv_device(){
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
}


