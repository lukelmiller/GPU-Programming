/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define ITERATION 100 // number of iteration
#define TILE_SIZE 14
#define BLOCK_SIZE 16

Matrix allocateMatrix(unsigned height, unsigned width);
void initMatrix(Matrix mat);
Matrix allocateDeviceMatrix(unsigned height, unsigned width);
void copyToDeviceMatrix(Matrix dst, Matrix src);
void copyFromDeviceMatrix(Matrix dst, Matrix src);
void verify(int *GPU_result, int *CPU_result, int height, int width);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void InitialGrid(int *grid, int height, int width);
void GiveLife(int flag,int n, int *grid,int  height, int width);
void GameofLife_CPU( int *grid, int width,int  height, int nowGrid);


#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
