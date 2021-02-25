#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void addKernel(int* c, int* a, int* b, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a[tid] + b[tid];
	//if (tid < n) {
	//	c[tid] = a[tid] + b[tid];
	//}
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* c, int* a, int* b, int size)
{
	// Launch a kernel on the GPU with one thread for each element.
	addKernel << < 1, size >> > (c, a, b, size);

}
