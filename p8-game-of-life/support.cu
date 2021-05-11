/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void InitialGrid(int *grid, int  height, int width)
{
        int i, j;
        for(i=0;i<height;i++)
                for(j=0;j<width;j++)
                        grid[0*width*height+i*width+j]=grid[1*width*height+i*width+j]=0;
}


void GiveLife(int flag,int n, int *grid,int  height, int width)
{
        int i;
        for(i=0;i<n;i++)
                grid[flag*height*width+(rand()%height)*width+rand()%width]=1;
}

Matrix allocateMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	mat.height = height;
	mat.width = mat.pitch = width;
	mat.elements = (float*)malloc(height*width*sizeof(float));
	if(mat.elements == NULL) FATAL("Unable to allocate host");

	return mat;
}

void initMatrix(Matrix mat)
{
    for (unsigned int i=0; i < mat.height*mat.width; i++) {
        mat.elements[i] = (rand()%100)/100.00;
    }
}

int IsLocationValid_CPU(int x, int y,int width, int height)
{
        if(x<0||y<0||x>=height||y>=width) return 0;
        else return 1;
}


int CountNeighbors_CPU(int flag,int x, int y, int  width, int height,int *grid)
{
        int count=0;
        int i, j;
        int range =3;
        for(i=-(range/2);i<=(range/2);i++)
        {
                for(j=-(range/2);j<=(range/2);j++)
                {
                        if(i==0&&j==0) continue;
                        if(IsLocationValid_CPU(x+i,y+j, width,height)==0) continue;
                        if(grid[flag*width*height+(x+i)*width +y+j]==1) count++;
                }
        }
        return count;
}

void GameofLife_CPU( int *grid, int width, int height, int nowGrid)
{
		 int count;
                 for(int i=0;i<height;i++)
                {
                        for(int j=0;j<width;j++)
                        {
                                count=CountNeighbors_CPU(nowGrid,i,j,width,height,grid);
                                if(grid[nowGrid*width*height+i*width+j]==0)
                                {
                                        if(count==3) grid[(1-nowGrid)*width*height+i*width+j]=1;
                                        else grid[(1-nowGrid)*width*height+i*width+j]=0;
                                }
                                else
                                {
                                        if(count<=1||count>=4) grid[(1-nowGrid)*width*height+i*width+j]=0;
                                        else grid[(1-nowGrid)*width*height+i*width+j]=1;
                                }
                        }
                }

}


Matrix allocateDeviceMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	cudaError_t cuda_ret;

	mat.height = height;
	mat.width = mat.pitch = width;
	cuda_ret = cudaMalloc((void**)&(mat.elements), height*width*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

	return mat;
}

void copyToDeviceMatrix(Matrix dst, Matrix src)
{
	cudaError_t cuda_ret;
	cuda_ret = cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
}

void copyFromDeviceMatrix(Matrix dst, Matrix src)
{
	cudaError_t cuda_ret;
	cuda_ret = cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy from device");
}

void verify(int *GPU_result, int *CPU_result, int height, int width) {

      for(int i=0;i<2*width*height; i++)
      {
	if(GPU_result[i]!=CPU_result[i])
	{
		printf("TEST FAILED\n\n");
        	exit(0);		
	}
      }			
		
  printf("TEST PASSED\n\n");

}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
	mat.elements = NULL;
}

void freeDeviceMatrix(Matrix mat)
{
	cudaFree(mat.elements);
	mat.elements = NULL;
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

