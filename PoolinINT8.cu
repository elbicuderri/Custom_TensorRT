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


__global__ void maxpooling_kernel_int8(char* output, char* input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width, int total_size, cudaStream_t stream)
{

	int C = channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = ((H + 2 * pH - kH) / sH) + 1;
	int Q = ((W + 2 * pW - kW) / sW) + 1;

	//tid : thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < total_size) {
		//q_idx : output w-index
		int q_idx = tid % Q;
		int idx = tid / Q;

		//p_idx : output h-index
		int p_idx = idx % P;
		idx /= P;

		//k_idx : output channel-index
		int k_idx = idx % C;

		//n_idx : output batch-index
		int n_idx = idx / C;

		//output(n_idx, k_idx, p_idx, q_idx)

		char max = -128;
		for (int kh = 0; kh < kH; kh++) {
			int h_idx = p_idx * sH + kh - pH;
			if (h_idx >= 0 && h_idx < H) {
				for (int kw = 0; kw < kW; kw++) {
					int w_idx = q_idx * sW + kw - pW;
					if (w_idx >= 0 && w_idx < W) {
						int input_index = n_idx * C * H * W + k_idx * H * W + h_idx * W + w_idx;
						if (input[input_index] > max) {
							max = input[input_index];
						}
					}
				}
			}
		}
		output[tid] = max;
	}

}


void maxpooling_int8(char* output, char* input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width, int pad_height, int pad_width, int stride_height, int stride_width, cudaStream_t stream)
{
	int N = batch;
	int C = channel;
	int H = height;
	int W = width;
	int kH = kernel_height;
	int kW = kernel_width;
	int pH = pad_height;
	int pW = pad_width;
	int sH = stride_height;
	int sW = stride_width;

	int P = (H + 2 * pH - kH) / sH + 1;
	int Q = (W + 2 * pW - kW) / sW + 1;

	int THREADS_PER_BLOCK = 256;
	int TOTAL_SIZE = N * C * P * Q;
	int NUMBER_OF_BLOCKS = (TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	maxpooling_kernel_int8 << < NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, 0, stream >> >
		(output, input, N, C, H, W, kH, kW, pH, pW, sH, sW, TOTAL_SIZE, stream);

}
