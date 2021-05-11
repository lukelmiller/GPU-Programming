/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *inout, unsigned size, float *sum) {
  // perform a local scan on 2*BLOCK_SIZE items
  __shared__ float sharedAry[BLOCK_SIZE * 2];
  unsigned int temp;
  int thread = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
  if (thread + BLOCK_SIZE < size) {
    sharedAry[threadIdx.x + BLOCK_SIZE] = inout[thread - 1 + BLOCK_SIZE];
  } else {
    sharedAry[threadIdx.x + BLOCK_SIZE] = 0;
  }
  if (thread < size && (threadIdx.x != 0 || blockIdx.x != 0)) {
    sharedAry[threadIdx.x] = inout[thread - 1];
  } else {
    sharedAry[threadIdx.x] = 0;
  }
  for (int i = 1; i < (BLOCK_SIZE * 2); i *= 2) {
    __syncthreads();
    temp = (threadIdx.x + 1) * 2 * i - 1;
    if (temp < (BLOCK_SIZE * 2)) {
      sharedAry[temp] += sharedAry[temp - i];
    }
  }
  for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    __syncthreads();
    temp = (threadIdx.x + 1) * 2 * i - 1;
    if (temp + i < BLOCK_SIZE * 2) {
      sharedAry[temp + i] += sharedAry[temp];
    }
  }
  __syncthreads();
  if (thread + BLOCK_SIZE < size) {
    inout[thread + BLOCK_SIZE] = sharedAry[threadIdx.x + BLOCK_SIZE];
  }
  if (thread < size) {
    inout[thread] = sharedAry[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    sum[blockIdx.x] = sharedAry[BLOCK_SIZE * 2 - 1];
  }
}

__global__ void addKernel(float *inout, float *sum, unsigned size) {
  // use the scan of partial sums to update 2*BLOCK_SIZE items
  int thread = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
  if (thread + BLOCK_SIZE < size) {
    inout[thread + BLOCK_SIZE] += sum[blockIdx.x];
  }
  if (thread < size) {
    inout[thread] += sum[blockIdx.x];
  }
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *inout, unsigned in_size) {
  float *sum;
  unsigned num_blocks;
  cudaError_t cuda_ret;
  dim3 dim_grid, dim_block;
  num_blocks = in_size / (BLOCK_SIZE * 2);
  if (in_size % (BLOCK_SIZE * 2) != 0)
    num_blocks++;
  dim_block.x = BLOCK_SIZE;
  dim_block.y = 1;
  dim_block.z = 1;
  dim_grid.x = num_blocks;
  dim_grid.y = 1;
  dim_grid.z = 1;
  cuda_ret = cudaMalloc((void **)&sum, num_blocks * sizeof(float));
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to allocate device memory");
  if (num_blocks > 1) {
    preScanKernel<<<dim_grid, dim_block>>>(inout, in_size, sum);
    preScan(sum, num_blocks);
    addKernel<<<dim_grid, dim_block>>>(inout, sum, in_size);
    cudaFree(sum);
  } else
    preScanKernel<<<dim_grid, dim_block>>>(inout, in_size, sum);
}