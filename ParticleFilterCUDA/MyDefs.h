#pragma once

#define PI 3.141592f
#define PI2 2.0f * PI
#define PI2SQRD sqrt(PI2)

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)

#define N 400'000
//#define N 1'000

#define ITERAIONS 20

//#define BLOCKSIZE 1024  // block dim 1D
#define BLOCKSIZE 512  // block dim 1D

#define MinX 0.0f
#define MaxX 50.0f

#define MinY 0.0f
#define MaxY 50.0f

#define MinHeading 0.0f
#define MaxHeading 3.0f
