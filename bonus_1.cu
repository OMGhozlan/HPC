#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <conio.h>


static cudaEvent_t start;
static cudaEvent_t finish;

void setupTimer(){
	cudaEventCreate(&start);
	cudaEventCreate(&finish);
	cudaEventRecord(start);
}

float getTime(){
	cudaEventRecord(finish);
	cudaEventSynchronize(finish);
	float time;
	cudaEventElapsedTime(&time, start, finish);
	cudaEventDestroy(start);
	cudaEventDestroy(finish);
	return time;
}

float* inv(float* M, int n){
	/*
	n -> number of rows and columns
	data -> array of float pointers with each array of dimension n x n where lda>=max(1,n)
	pivots -> pivoting sequence
	info -> factorization info / inversion info
	d2 -> leading dimnsion of 2d array ised to store each matrix of pivots[i]
	*/
	cublasHandle_t handle;
	cublasCreate(&handle);
	float **data, **d2;
	float *dL, *dC;
	int *pivots, *info;
	size_t s = n * n * sizeof(float);
	cudaMalloc(&data, sizeof(float*));
	cudaMalloc(&d2, sizeof(float*));
	cudaMalloc(&dL, s);
	cudaMalloc(&dC, s);
	cudaMalloc(&pivots, n * sizeof(int));
	cudaMalloc(&info, sizeof(int));
	cudaMemcpy(dL, M, s, cudaMemcpyHostToDevice);
	cudaMemcpy(data, &dL, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d2, &dC, sizeof(float*), cudaMemcpyHostToDevice);
	setupTimer();
	cublasSgetrfBatched(handle, n, data, n, pivots, info, 1);
	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, n, (const float **)data, n, pivots, d2, n, info, 1);
	cudaDeviceSynchronize();
	float time = getTime();
	printf("cuBLAS inverse in: %.3f ms.\n", time);
	float* res = (float*)malloc(s);
	cudaMemcpy(res, dC, s, cudaMemcpyDeviceToHost);
	cudaFree(data);
	cudaFree(d2);
	cudaFree(dL);
	cudaFree(dC);
	cudaFree(pivots);
	cudaFree(info);
	cublasDestroy(handle);
	return res;
}

__host__ __device__ unsigned int getIndex(unsigned int i, unsigned  int j, unsigned int ld){
	return j*ld + i; 
}

__device__ float det(float *b, unsigned int *n, cublasHandle_t *hdl){
	int *info = (int *)malloc(sizeof(int)); 
	info[0] = 0;
	int batch = 1; 
	int *p = (int *)malloc(*n*sizeof(int));
	float **a = (float **)malloc(sizeof(float *));
	*a = b;
	cublasStatus_t status = cublasSgetrfBatched(*hdl, *n, a, *n, p, info, batch);
	float p_res = 1.0f;
	for (int i = 0; i<(*n); ++i)
		p_res *= b[getIndex(i, i, *n)];
	return p_res;
}

__global__ void runtest(float *a_i, unsigned int n){
	cublasHandle_t handle;
	cublasCreate(&handle);
	printf("det on GPU:%f\n", det(a_i, &n, &hdl));
	cublasDestroy(handle);
}

int main(int argc, char** argv){
	int n = 1000;
	float* M = (float*)malloc(n * n * sizeof(float));
	for (int i = 0; i < n * n; i++)
		M[i] = ((float)rand() / (float)(RAND_MAX));
	float* i = inv(M, n);
	printf("Finished.");
	return 0;
}