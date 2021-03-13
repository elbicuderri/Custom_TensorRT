#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <cassert>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "EntropyCalibrator.h"

#include <cstdlib>
#include <cstring>
#include <time.h>
#include <string>
#include <algorithm>
#include <vector>
#include <limits.h>
#include <float.h>

#include "ConvPlugin.h"

void conv_plugin_func(float* output, float* input, float *weight, float *bias,
	int batch, int out_channel, Dims dim_input, Dims dim_kernel,
	Dims dim_pad, Dims dim_stride, cudaStream_t stream);

ConvPluginV2::ConvPluginV2(ITensor* data, int BatchSize, int oC, int iC, int iH, int iW,
	int kH, int kW, int pH, int pW, int sH, int sW,
	Weights Weight, Weights Bias)
	: iType(data->getType())
	, N(BatchSize)
	, iC(iC)
	, iH(iH)
	, iW(iW)
	, oC(oC)
	, oH(((iH - 2 * pH - kH) / sH) + 1)
	, oW(((iW - 2 * pW - kW) / sW) + 1)
	, kH(kH)
	, kW(kW)
{
	h_weight.insert(h_weight.end(), &((float*)Weight.values)[0], &((float*)Weight.values)[oC*iC*kH*kW]); // Weights 타입을 직접 이용한 방법.
	h_bias.insert(h_bias.end(), &((float*)Bias.values)[0], &((float*)Bias.values)[oC]);
}

ConvPluginV2::~ConvPluginV2()
{
}

int ConvPluginV2::getNbOutputs() const { return 1; }

int ConvPluginV2::initialize() { return 0; }

void ConvPluginV2::terminate() {}

Dims ConvPluginV2::getOutputDimensions(int index,
	const Dims* inputs, int npInputDims)
{
	return Dims{ 3, {oC, oH, oW} };
}

size_t ConvPluginV2::getWorkspaceSize(int maxBatchSize) const { return 0; }

/*
cudastream을 이용한 cudamemcpyasync 구현.
*/
int ConvPluginV2::enqueue(int batchSize, const void* const* inputs,
	void** outputs, void* workspace, cudaStream_t stream)
{
	float* d_Weight{ nullptr };

	CHECK(cudaMalloc((void**)&d_Weight, oC * iC * kH * kW * sizeof(float)));

	cudaMemcpyAsync((void*)d_Weight, (const void*)&h_weight[0],
		oC * iC * kH * kW * sizeof(float), cudaMemcpyHostToDevice, stream);

	float* d_Bias{ nullptr };

	CHECK(cudaMalloc((void**)&d_Bias, oC * sizeof(float)));

	cudaMemcpyAsync((void*)d_Bias, (const void*)&h_bias[0], oC * sizeof(float), cudaMemcpyHostToDevice, stream);

	conv_plugin_func((float*)outputs[0], (float*)inputs[0],
		(float*)d_Weight, (float*)d_Bias, N, oC,
		Dims{ 3, {iC, iH, iW} },
		Dims{ 2, {kH, kW} },
		Dims{ 2, {0,0} },
		Dims{ 2, {1,1} },
		stream);

	return 0;
}

size_t ConvPluginV2::getSerializationSize() const
{
	// iC, iH, iW, oC, oH, oW
	return 6 * sizeof(int);
}

template<typename T> void write(char*& buffer, const T& val)
{
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

void ConvPluginV2::serialize(void* buffer) const
{
	char *d = reinterpret_cast<char*>(buffer), *a = d;
	write(d, iC);
	write(d, iH);
	write(d, iW);
	write(d, oC);
	write(d, oH);
	write(d, oW);
	assert(d == a + getSerializationSize());
}

void ConvPluginV2::configurePlugin(const Dims* inputDims, int nbInputs,
	const Dims* outputDims, int nbOutputs,
	const DataType* inputTypes, const DataType* outputTypes,
	const bool* inputIsBroadcast, const bool* outputIsBroadcast,
	PluginFormat floatFormat, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(nbOutputs == 1);

	iC = inputDims->d[0];
	iH = inputDims->d[1];
	iW = inputDims->d[2];

	oC = outputDims->d[0];
	oH = outputDims->d[1];
	oW = outputDims->d[2];

	iType = inputTypes[0];
}

bool ConvPluginV2::supportsFormat(DataType type, PluginFormat format) const
{
	return ((type == DataType::kFLOAT || type == DataType::kHALF)
		&& format == PluginFormat::kNCHW);
}

const char* ConvPluginV2::getPluginType() const
{
	return "ConvPluginV2";
}

const char* ConvPluginV2::getPluginVersion() const
{
	return "2";
}

void ConvPluginV2::destroy()
{
	delete this;
}

IPluginV2Ext* ConvPluginV2::clone() const
{
	auto* plugin = new ConvPluginV2(*this);
	return plugin;
}

DataType ConvPluginV2::getOutputDataType(int index, const DataType* inputTypes,
	int nbInputs) const
{
	return inputTypes[0];
}

void ConvPluginV2::setPluginNamespace(const char* pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char* ConvPluginV2::getPluginNamespace() const
{
	return mPluginNamespace;
}

bool ConvPluginV2::isOutputBroadcastAcrossBatch(int outputIndex,
	const bool* inputIsBroadcasted, int nbInputs) const
{
	return false;
}

bool ConvPluginV2::canBroadcastInputAcrossBatch(int inputIndex) const
{
	return false;
}