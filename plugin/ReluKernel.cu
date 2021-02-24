#include <math.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "NvInfer.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <float.h>
#include <cfloat>

__global__ void relu_plugin_kernel(float *input, int total_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	if (input[tid] < 0.0f) 
    {
		input[tid] = 0.0f;
	}

}

void relu_plugin(float *input, int batch, int channel, int height, int width, cudaStream_t stream)
{

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = batch * channel * height * width;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	relu_plugin_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream >> > (input, TOTAL_SIZE);
}

__global__ void relu_plugin_int8_kernel(char* input, int total_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_size)
		return;

	if (input[tid] < 0) 
    {
		input[tid] = 0;
	}

}

void relu_plugin_int8(char* input, int batch, int channel, int height, int width, cudaStream_t stream)
{

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = batch * channel * height * width;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	relu_plugin_int8_kernel << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream >> > (input, TOTAL_SIZE);
}