
/* select-
 *         0: input in the first half, output in the second half
 *         1: input in the second half, output in the first half
 */

#include "support.h"

__global__ void GameofLife(int *GPUgrid, int select, int width, int height) {

  // Insert code here...

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  // Index (T=Top, M=Middle, B=Bottom, I=Index)
  int TLI, TI, TRI, MLI, MI, MRI, BLI, BI, BRI, OUTI, neighSum;
  TLI = TI = TRI = MLI = MI = MRI = BLI = BI = BRI = OUTI = neighSum = 0;

  int gridRadius = height * width;

  if ((row >= 0) && (row < height) && (col >= 0) && (col < width)) {
    if (select == 0) {
      if (row == 0)
        MI = col;
      else
        MI = (row - 1) * width + col;
      MLI = MI - 1;
      MRI = MI + 1;
      TI = MI - width;
      TLI = TI - 1;
      TRI = TI + 1;
      BI = MI + width;
      BLI = BI - 1;
      BRI = BI + 1;
      OUTI = row * width + col + gridRadius;
    } else {
      if (row == 0)
        MI = col + gridRadius;
      else
        MI = (row - 1) * width + col + gridRadius;
      MLI = MI - 1 + gridRadius;
      MRI = MI + 1 + gridRadius;
      TI = MI - width + gridRadius;
      TLI = TI - 1 + gridRadius;
      TRI = TI + 1 + gridRadius;
      BI = MI + width + gridRadius;
      BLI = BI - 1 + gridRadius;
      BRI = BI + 1 + gridRadius;
      OUTI = MI - gridRadius;
    }
    if (row == 0 && col == 0) {
      neighSum = GPUgrid[BI] + GPUgrid[BRI] + GPUgrid[MRI];
    } else if (row == height && col == width) {
      neighSum = GPUgrid[TI] + GPUgrid[TLI] + GPUgrid[MLI];
    } else if (row == 0 && col == width) {
      neighSum = GPUgrid[BI] + GPUgrid[BLI] + GPUgrid[MLI];
    } else if (row == height && col == 0) {
      neighSum = GPUgrid[TI] + GPUgrid[TRI] + GPUgrid[MRI];
    } else if (row == 0) {
      neighSum = GPUgrid[BI] + GPUgrid[BRI] + GPUgrid[MRI] + GPUgrid[TRI] +
                 GPUgrid[TI];
    } else if (col == 0) {
      neighSum = GPUgrid[BI] + GPUgrid[BRI] + GPUgrid[MRI] + GPUgrid[MLI] +
                 GPUgrid[BLI];
    } else {
      neighSum = GPUgrid[TLI] + GPUgrid[TI] + GPUgrid[TRI] + GPUgrid[MLI] +
                 GPUgrid[MRI] + GPUgrid[BLI] + GPUgrid[BI] + GPUgrid[BRI];
    }
    __syncthreads();
    if (neighSum < 2)
      GPUgrid[OUTI] = 0;
    else if (neighSum == 3)
      GPUgrid[OUTI] = 1;
    else if (neighSum > 3)
      GPUgrid[OUTI] = 0;
  }
}
