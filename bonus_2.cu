#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <conio.h>

/*
void printMatrix(float *M, int n){
	int i, j;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			//printf("%.2f\t", M[i][j]);
			printf("%.2f\t", M[(i * n) + j]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

void LUDecompose(float* A, float* L, float* U, int n){
	float sum = 0.0f;
	for (int i = 0; i < n; i++) { 
		for (int k = i; k < n; k++) { //Upper Trianglular
			sum = 0.0f;
			for (int j = 0; j < i; j++)
				sum += (L[(i* n) + j] * U[(j* n) + k]);
			U[(i * n) + k] = A[(i* n) + k] - sum;
		}

		for (int k = i; k < n; k++) { //Lower Trianglular
			if (i == k)
				L[(i* n) + i] = 1; // Diagonal as 1
			else {
				sum = 0.0f;
				for (int j = 0; j < i; j++)
					sum += (L[(k* n) + j] * U[(j* n) + i]);
				L[(k* n) + i] = (A[(k* n) + i] - sum) / U[(i* n) + i];
			}
		}
	}
}
int main(int argc, char** argv){
	int n = 4;
	float *A = (float *)malloc(n * n * sizeof(float));
	float *L = (float *)malloc(n * n * sizeof(float));
	float *U = (float *)malloc(n * n * sizeof(float));
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			A[(i * n) + j] = (float)(1 + (rand() % 10)); //n => column~width
			L[(i * n) + j] = 0.0f;
			U[(i * n) + j] = 0.0f;
		}
	}
	LUDecompose(A, L, U, n);
	printMatrix(A, n);
	printMatrix(L, n);
	printMatrix(U, n);
	return 0;
}

*/

__device__ float getLowerAt(float *M, unsigned int y, unsigned int x, unsigned int width){
	if (y < x) return 0.0f;
	if (y == x) return 1.0f;
	return M[y * width + x];
}

__device__ float getUpperrAt(float *M, unsigned int y, unsigned int x, unsigned int width){
	if (y > x) return 0.0;
	return M[y * width + x];
}

__global__ void invertUpper(float *dest, float *M, unsigned int width, unsigned int i){
	unsigned int y, x;
	y = -blockIdx.y * blockDim.y - threadIdx.y;
	x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y == 0){
		if (x == 0){
			dest[i * width + i] = 1 / M[i * width + i];
			return;
		}
		dest[i * width + i + x] = M[i * width + i + x] * M[i * width + i];
		return;
	}

	if (x == 0){
		dest[(i + y) * width + i] = -(M[(i + y)*width + i] * M[i*width + i]);
		return;
	}
	dest[(i + y) * width + x + i] = M[(i + y) * width + x + i] + M[(i + y) * width + i] * M[i * width + i + x];
}

__global__ void invertLower(float *dest, float *M, unsigned int width, unsigned int i){
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = -blockIdx.x * blockDim.x - threadIdx.x;

	if (y == 0){
		if (x == 0)
			return;
		dest[i * width + i + x] = M[i * width + i + x];
		return;
	}

	if (x == 0){
		dest[(i + y) * width + i] = -M[(i + y) * width + i];
		return;
	}
	dest[(i + y) * width + x + i] = M[(i + y) * width + x + i] + M[(i + y) * width + i] * M[i * width + i + x];
}

__global__ void toLULCol(float *M, unsigned int width){
	unsigned int pos = width * (blockIdx.y * blockDim.y + threadIdx.y + 1);
	M[pos] /= M[0];
}

__global__ void toLURow(float *M, unsigned int y, unsigned int initPos, unsigned int width){
	unsigned int i, x = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = 0; i<y; i++)
		M[initPos + x] -= M[y*width + i] * M[i*width + x + y];
}

__global__ void multiply(float *dest, float *M, float *N, unsigned int width){
	unsigned int i, pos, x, y;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	pos = y * width + x;

	dest[pos] = M[y*width] * N[x];
	for (i = 1; i<width; i++)
		dest[pos] += M[y * width + i] * N[i * width + x];
}
__global__ void multiplyLU(float *dest, float *lu, unsigned int width){
	unsigned int i, pos, x, y;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	pos = y * width + x;

	dest[pos] = getLowerAt(lu, y, 0, width) * getUpperrAt(lu, 0, x, width);
	for (i = 1; i<width; i++)
		dest[pos] += getLowerAt(lu, y, i, width) * getUpperrAt(lu, i, x, width);
}

__global__ void multiplyUL(float *dest, float *lu, unsigned int width){
	unsigned int i, pos, x, y;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	pos = y * width + x;

	dest[pos] = getUpperrAt(lu, y, 0, width) * getLowerAt(lu, 0, x, width);
	for (i = 1; i<width; i++)
		dest[pos] += getUpperrAt(lu, y, i, width) * getLowerAt(lu, i, x, width);
}

__global__ void _matDifferent(float *M, float *N, unsigned int width, const float tolerance, bool *result){
	unsigned int pos, x, y;
	x = blockIdx.x*blockDim.x + threadIdx.x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	pos = y * width + x;
	float diff = M[pos] - N[pos];
	if (diff < -tolerance || diff > tolerance)
		*result = true;
}

typedef struct {
	float *h; /*Host data*/
	float *d; /*Device data*/
	unsigned int w; /*Matrix size*/
	bool tapped; /*Has been multiplied*/
	bool isInLU; /*Decomposed into L and U matricies*/
}Matrix;

/*Copy matrix from device to host*/
void copyHtoD(Matrix M){
	cudaMemcpy(M.d, M.h, M.w * M.w *sizeof(float), cudaMemcpyHostToDevice);
	M.tapped = 0;
}
/*Copy matrix from host to device*/
void copyDtoH(Matrix M){
	cudaMemcpy(M.h, M.d, M.w * M.w *sizeof(float), cudaMemcpyDeviceToHost);
	M.tapped = 0;
}


/*Create matrix*/
void init(Matrix M, unsigned int width){
	M.isInLU = false;
	M.tapped = false;
	M.w = width;
	size_t size = M.w * M.w * sizeof(float);
	cudaMalloc((void **)&M.d, size);
	M.h = (float *)malloc(size);
	if (!M.h){
		cudaFree(M.d);
		printf("Host array unalloacted or null\n");
	}
}

/*Fill matrix with random values*/
void fill(Matrix M){
	unsigned int i;
	for (i = 0; i< M.w * M.w; i++)
		M.h[i] = (float)(rand() % 128) + 1.0f;
}

void print(Matrix M){
	unsigned int x, y;
	for (x = 0; x < M.w; x++){
		for (y = 0; y < M.w; y++){
			printf("%.2f\t", M[(x * M.w) + y]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

/*Copy matrix*/
Matrix copy(Matrix M){
	Matrix temp;
	size_t size = M.w * M.w * sizeof(float);
	temp.w = M.w;
	memcpy((void *)M.h, (const void *)M.h, size);
	cudaMemcpy(temp.d, M.d, size, cudaMemcpyDeviceToDevice);
	temp.isInLU = M.isInLU;
	return temp;
}

/*Free matrix data*/
void free(Matrix M){
	free(M.h);
	cudaFree(M.d);
}

/*Decompse M as L and U where M = LU*/
void toLU(Matrix M){
	unsigned int x, y, p;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(1, M.w - 1);
	toLULCol << <dimGrid, dimBlock >> >(M.d, M.w);
	copyDtoH(M);
	dimBlock = dim3(M.w - 1, 1);
	toLURow << <dimGrid, dimBlock >> >(M.d, 1, M.w + 1, M.w);
	/*Update host*/
	size_t size = (M.w - 1) * sizeof(float);
	cudaMemcpy(&M.h[M.w + 1], &M.d[M.w + 1], size, cudaMemcpyDeviceToHost);

	for (y = 2; y < M.w; y++){
		for (x = 1; x < y; x++){
			for (p = 0; p < x; p++)
				M.h[y * M.w + x] -= M.h[y * M.w + p] * M.h[p * M.w + x];
			M.h[y * M.w + x] /= M.h[x * M.w + x];
		}
		/*Update host*/
		size = (y - 1)*sizeof(float);
		cudaMemcpy(&M.d[y * M.w + 1], &M.h[y * M.w + 1], size, cudaMemcpyHostToDevice);
		dimBlock = dim3(M.w - y, 1);
		toLURow << <dimGrid, dimBlock >> >(M.d, y, y * M.w + y, M.w);
		/*Update host*/
		size = (M.w - y)*sizeof(float);
		cudaMemcpy(&M.h[y * M.w + y], &M.d[y * M.w + y], size, cudaMemcpyDeviceToHost);
	}

	M.isInLU = true;
}

/*Get inverse of L and U*/
void invLU(Matrix M){
	unsigned int i;
	size_t size = M.w * M.w *sizeof(float);
	float *dest, *temp = (float *) malloc(size);
	memcpy((void *)temp, (const void *)M.h, size);
	dest = M.h;
	cudaMalloc((void **)&temp, size);
	for(i=0; i < M.w; i++){
		int min = (M.w - i < i + 1 ? M.w - i : i + 1);
		dim3 dimGrid(1, 1);
		dim3 dimBlock(min, min);
		invertUpper << <dimGrid, dimBlock >> >(temp, M.d, M.w, i);
		invertLower << <dimGrid, dimBlock >> >(temp, M.d, M.w, i);
	}
	cudaMemcpy(M.d, temp, size, cudaMemcpyDeviceToDevice);
	copyDtoH(M);
	cudaFree(temp);
}

/*Multiply matrix A and B to get C*/
Matrix mult(Matrix M, Matrix N){
	Matrix temp;
	if (M.w != N.w){
		printf("Matrices must be of the same size\n");
		return temp;
	}
	init(temp, M.w);
	dim3 dimGrid(1, 1);
	dim3 dimBlock(temp.w, temp.w);
	multiply << <dimGrid, dimBlock >> >(temp.d, M.d, N.d, temp.w);
	temp.isInLU = false;
	temp.tapped = true;
	return temp;
}

void multiplyUL(Matrix M){
	float *temp;
	size_t size = M.w * M.w * sizeof(float);
	dim3 dimGrid(1, 1);
	dim3 dimBlock(M.w, M.w);
	cudaMalloc((void **)&temp, size);
	cudaMemcpy(temp, M.d, size, cudaMemcpyDeviceToDevice);
	multiplyUL << <dimGrid, dimBlock >> >(M.d, temp, M.w);
	cudaFree(temp);
	M.isInLU = false;
	M.tapped = true;
}

int main(int argc, char **argv){
	Matrix M, LU, INV;
	unsigned int width;
	
	printf("Matrix width and height: %u\n", width);

	printf("Allocating memory...\n");
	init(M, width);
	printf("Filling matrix with random numbers...\n");
	fill(M);
	printf("Matrix\n");
	print(M);
	printf("Decomposing in LU...\n");
	LU = copy(M);
	toLU(LU);
	print(LU);
	INV = copy(LU);
	printf("Inverting the LU...\n");
	invLU(INV);
	print(INV);
	//multiplyUL(INV);
	printf("Inverse = Inv(U) * Inv(L):\n");
	print(INV);
	return 0;
}