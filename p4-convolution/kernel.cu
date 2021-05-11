/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
 #include "support.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];


// M: filter, N: input image, P: output image
__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE
	
 	__shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	
	int row_i = row_o - 2;  // Assumes kernel size is 5
	int col_i = col_o - 2;  //Assumes kernel size is 5

	if((row_i >= 0) && (row_i < N.height) && (col_i >= 0)  && (col_i < N.width) )
    	N_s[ty][tx] = N.elements[row_i*N.width + col_i];
  	else
    	N_s[ty][tx] = 0.0f;

	__syncthreads();

	float output = 0.0f;
	if(ty < TILE_SIZE && tx < TILE_SIZE){
		for(int i = 0; i < FILTER_SIZE; i++)
		  for(int j = 0; j < FILTER_SIZE; j++)
			output += M_c[i][j] * N_s[i+ty][j+tx];
		
		if(row_o < P.height && col_o < P.width)
  			P.elements[row_o * P.width + col_o] = output;
	}

	



















}
