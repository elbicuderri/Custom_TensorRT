#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <float.h>

__global__ void relu_kernel(float *output, float *input, int total_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	if (input[tid] > 0.0f) {
		output[tid] = input[tid];
	}
	else {
		output[tid] = 0.0f;
	}
}

void relu(float *output, float *input, int batch, int channel, int height, int width, cudaStream_t stream)
{

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = batch * channel * height * width;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	relu_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream >> > (output, input, TOTAL_SIZE);
}

__global__ void relu_int8_kernel(char* output, char* input, int total_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	if (input[tid] > 0) {
		output[tid] = input[tid];
	}
	else {
		output[tid] = 0;
	}
}

void relu_int8(char* output, char* input, int batch, int channel, int height, int width, cudaStream_t stream)
{

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = batch * channel * height * width;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	relu_int8_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream >> > (output, input, TOTAL_SIZE);
}