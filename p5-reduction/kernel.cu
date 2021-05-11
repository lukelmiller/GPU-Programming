/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float outArr[2*BLOCK_SIZE];

    int tx = threadIdx.x;
    int i = (2* blockIdx.x * blockDim.x) + tx;

    outArr[tx] = 0.0;
    if(i < size){
        outArr[tx] = in[i];
    }
    outArr[BLOCK_SIZE + tx] = 0.0;
    if(i + BLOCK_SIZE < size)
        outArr[BLOCK_SIZE + tx] = in[i + BLOCK_SIZE];
    __syncthreads();
    

    for (int offset = BLOCK_SIZE; offset > 0; offset >>= 1) {
        if (tx < offset) 
            outArr[tx] += outArr[tx + offset];
        __syncthreads();
    }

    if(tx == 0)
        out[blockIdx.x] = outArr[0];
    __syncthreads();
    


}
