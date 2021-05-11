#include <stdio.h>
#include <iostream>
#include "support.h"
#include "kernel.cu"



int main(int argc, char* argv[])
{
    Timer timer;

    time_t t;
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    cudaError_t cuda_ret;
    int  height, width;


	/* Read image dimensions */
    if (argc == 1) {
        height = 1400;
	width =1400;
    } else if (argc == 2) {
        height= atoi(argv[1]);
	width= atoi(argv[1]);
    } else if (argc == 3) {
        height = atoi(argv[1]);
        width = atoi(argv[2]);
    }else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./gameoflife          # Matrix is 1400 x 1400"
           "\n    Usage: ./gameoflife <m>      # Matrix is m x m"
	   "\n    Usage: ./gameoflife <m> <n>  # Matrix is m x n"
           "\n");
        exit(0);
    }
    
    
    
	/* Allocate host memory */
	int *grid=new int [height*width*2];
	int *Ggrid_result=new int [height*width*2];
	/* Initialize Matrix */
	InitialGrid(grid,height,width);
	GiveLife(0,height*width/2,grid,height,width);


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    printf("\nThe size of the universe is %d x %d.\n\n", height, width);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
    int *GPUgrid;

    long long int size=sizeof(int)*2*width*height;
    cuda_ret = (cudaMalloc((void**) &GPUgrid, size));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate GPU global memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret =(cudaMemcpy(GPUgrid,grid,size,cudaMemcpyHostToDevice));
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy to constant memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    
    int h_blocks = height/TILE_SIZE ;
    int w_blocks = width/TILE_SIZE ;

    dim3 dim_grid(w_blocks,h_blocks);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    
    // INSERT CODE ABOVE
   
	int select =0;
	for(int m=0;m<ITERATION;m++){
        	GameofLife<<<dim_grid, dim_block>>>(GPUgrid,select,width,height);
         	select=1-select;
        }
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...\n"); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(Ggrid_result,GPUgrid,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

   //CPU -----------------------------------------------------------------------

	int nowGrid=0;
        for(int n=0;n<ITERATION;n++)
        {
		GameofLife_CPU( grid, width, height,nowGrid);
		nowGrid=1-nowGrid;
		
      	}

// Verify correctness -----------------------------------------------------
	printf("Verifying..."); fflush(stdout);
	verify(Ggrid_result,grid,height,width);

// Free memory ------------------------------------------------------------
	 cudaFree(GPUgrid);

	 delete [] grid;		
	 delete [] Ggrid_result;
	 return 0;
}
