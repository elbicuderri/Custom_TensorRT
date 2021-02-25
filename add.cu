#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

__global__ void addKernel(int *c, int *a, int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, int *a, int *b, int size)
{

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (c, a, b);

}
