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


    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    //flushes the output buffer 
    fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        //converts string argv into int
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    
    
    /* Intializes First Array w/ n random numbers */
    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) 
    { 
        A_h[i] = (rand()%100)/100.00; 
    }

    /* Intializes second Array w/ n random numbers */
    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) 
    { 
        B_h[i] = (rand()%100)/100.00; 
    }

    /* Intializes final Array w/ n slots */
    float* C_h = (float*) malloc( sizeof(float)*n );

    //Outputs time taken to intialize host variables
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    float* A_d;
    cudaMalloc(&A_d, sizeof(float)*n );

    float* B_d;
    cudaMalloc(&B_d, sizeof(float)*n );
    
    float* C_d;
    cudaMalloc(&C_d, sizeof(float)*n );

    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cudaMemcpy(A_d, A_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*n, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);



    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); 
    fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cudaMemcpy(C_h, C_d, sizeof(float)*n, cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); 
    fflush(stdout);
    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);



    return 0;

}

