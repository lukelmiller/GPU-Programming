/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <time.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0); 
    cudaStreamCreate(&stream1);
    const unsigned int THREADS_PER_BLOCK = 256;
    

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    const unsigned int numBlocks = (n - 1)/THREADS_PER_BLOCK + 1;
    dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    

    float* A_h0 = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h0[i] = (rand()%100)/100.00; }

    float* B_h0 = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h0[i] = (rand()%100)/100.00; }

    float* C_h0 = (float*) malloc( sizeof(float)*n );

    float* A_h1 = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h1[i] = (rand()%100)/100.00; }

    float* B_h1 = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h1[i] = (rand()%100)/100.00; }

    float* C_h1 = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    float* A_d0;
    cuda_ret = cudaMalloc((void**) &A_d0, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* B_d0;
    cuda_ret = cudaMalloc((void**) &B_d0, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* C_d0;
    cuda_ret = cudaMalloc((void**) &C_d0, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* A_d1;
    cuda_ret = cudaMalloc((void**) &A_d1, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* B_d1;
    cuda_ret = cudaMalloc((void**) &B_d1, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* C_d1;
    cuda_ret = cudaMalloc((void**) &C_d1, sizeof(float)*n);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Without Streams ----------------------------------------------
    printf("Running WITHOUT streams.."); fflush(stdout);
    startTime(&timer);

    for (int i=0; i<1000000; i+=1) {

        //COPY MEMORY
        cuda_ret = cudaMemcpy(A_d0, A_h0, sizeof(float)*n, cudaMemcpyHostToDevice);
	    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cuda_ret = cudaMemcpy(B_d0, B_h0, sizeof(float)*n, cudaMemcpyHostToDevice);
	    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cuda_ret = cudaMemcpy(A_d1, A_h1, sizeof(float)*n, cudaMemcpyHostToDevice);
	    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cuda_ret = cudaMemcpy(B_d1, B_h1, sizeof(float)*n, cudaMemcpyHostToDevice);
	    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cudaDeviceSynchronize(); 

        //LAUNCH KERNEL
        vecAddKernel<<< gridDim, blockDim >>> (A_d0, B_d0, C_d0, n);
        cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
        vecAddKernel<<< gridDim, blockDim >>> (A_d1, B_d1, C_d1, n);
        cuda_ret = cudaDeviceSynchronize();
	    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");

        //COPY RESULTS
        cuda_ret = cudaMemcpy(C_h0, C_d0, sizeof(float)*n, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
        cuda_ret = cudaMemcpy(C_h1, C_d1, sizeof(float)*n, cudaMemcpyDeviceToHost);
	    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");
        cudaDeviceSynchronize();
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // Verify WITHOUT streams correctness -----------------------------------------
    printf("Verifying WITHOUT streams results 0..."); fflush(stdout);
    verify(A_h0, B_h0, C_h0, n);
    printf("Verifying WITHOUT streams results 1..."); fflush(stdout);
    verify(A_h1, B_h1, C_h1, n);



    // With Streams ----------------------------------------------
    printf("Running WITH streams.."); fflush(stdout);
    startTime(&timer);

    for (int i=0; i<1000000; i+=1) {
        // Copy host variables to device ------------------------------------------
        cuda_ret = cudaMemcpyAsync(A_d0, A_h0, n*sizeof(float),cudaMemcpyHostToDevice, stream0); 
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device"); 
        cuda_ret = cudaMemcpyAsync(B_d0, B_h0, n*sizeof(float),cudaMemcpyHostToDevice, stream0); 
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        vecAddKernel<<< gridDim, blockDim, 0, stream0 >>> (A_d0, B_d0, C_d0, n);
        cuda_ret = cudaMemcpyAsync(A_d1, A_h1, n*sizeof(float),cudaMemcpyHostToDevice, stream1); 
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cuda_ret = cudaMemcpyAsync(B_d1, B_h1, n*sizeof(float),cudaMemcpyHostToDevice, stream1);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cudaStreamSynchronize(stream0);
        
        vecAddKernel<<< gridDim, blockDim, 0, stream1>>> (A_d1, B_d1, C_d1, n);

        // Copy device variables from host ----------------------------------------
        cuda_ret = cudaMemcpyAsync(C_h0, C_d0, n*sizeof(float),cudaMemcpyDeviceToHost, stream0);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
        cudaStreamSynchronize(stream1);
        cuda_ret = cudaMemcpyAsync(C_h1, C_d1, n*sizeof(float),cudaMemcpyDeviceToHost, stream1); 
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
    }
    
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    

    // Verify WITH streams correctness --------------------------------------------
    printf("Verifying WITH streams results 0..."); fflush(stdout);
    verify(A_h0, B_h0, C_h0, n);
    printf("Verifying WITH streams results 1..."); fflush(stdout);
    verify(A_h1, B_h1, C_h1, n);


    // Free memory ------------------------------------------------------------

    free(A_h0);
    free(B_h0);
    free(C_h0);
    free(A_h1);
    free(B_h1);
    free(C_h1);

    cuda_ret = cudaFree(A_d0);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");
	
    cuda_ret = cudaFree(B_d0);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");

    cuda_ret = cudaFree(C_d0);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");
    
    cuda_ret = cudaFree(A_d1);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");
	
    cuda_ret = cudaFree(B_d1);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");

    cuda_ret = cudaFree(C_d1);
	if(cuda_ret != cudaSuccess) FATAL("Unable to free CUDA memory");

    return 0;

}

