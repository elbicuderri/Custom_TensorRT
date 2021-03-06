//#pragma once
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

void relu(float *output, float *input, int batch, int channel, int height, int width, cudaStream_t stream);

void relu_int8(char* output, char* input, int batch, int channel, int height, int width, cudaStream_t stream);

class ReluPluginV2IO : public nvinfer1::IPluginV2IOExt
{
public:
	DataType iType;
	int N, C, H, W;
	std::string mPluginNamespace;
	std::string mNamespace;

public:
	ReluPluginV2IO() = delete;

	ReluPluginV2IO(ITensor* data, int BatchSize, int ic, int ih, int iw)
		: N(BatchSize)
		, C(ic)
		, H(ih)
		, W(iw)
	{
		iType = data->getType();
	}

	ReluPluginV2IO(ITensor& data, int BatchSize, int ic, int ih, int iw)
		: N(BatchSize)
		, C(ic)
		, H(ih)
		, W(iw)
	{
		iType = data.getType();
	}

	~ReluPluginV2IO() override
	{
		std::cout << "plugin disapper~" << std::endl;
	}

	int getNbOutputs() const override { return 1; }

	int initialize() override { return 0; }

	void terminate() override {}

	Dims getOutputDimensions(int index,
		const Dims* inputs, int npInputDims) override
	{
		return Dims{ 3, {C, H, W} };
	}

	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

	int enqueue(int batchSize, const void* const* inputs,
		void** outputs, void* workspace, cudaStream_t stream) override
	{

		if (iType == DataType::kINT8)
		{
			relu_int8((char*)outputs[0], (char*)inputs[0],
				N, C, H, W, stream);
		}

		//else if ((iType == DataType::kHALF)
		//{
		//	maxpooling_half((half_float::half*)outputs[0], (half_float::half*)inputs[0],
		//		N, iC, iH, iW, kH, kW, pH, pW, sH, sW, stream);
		//}

		else
		{
			relu((float*)outputs[0], (float*)inputs[0],
				N, C, H, W, stream);
		}

		return 0;
	}

	size_t getSerializationSize() const override
	{
		return 0;
	}

	void serialize(void* buffer) const override
	{
	}

	void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
	{
		assert(in && nbInput == 1);
		assert(out && nbOutput == 1);
		assert(in[0].type == out[0].type);
		iType = in[0].type; // iType is changed when in[0].type is changed.
		if (iType == DataType::kINT8)
		{
			std::cout << "Pooling Type is INT8" << std::endl;
		}

		else
		{
			std::cout << "Pooling Type is FLOAT32" << std::endl;
		}
		assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
	}


	bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
		int32_t nbInputs, int32_t nbOutputs) const override
	{
		assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
		bool condition = inOut[pos].format == TensorFormat::kLINEAR;
		condition &= inOut[pos].type != DataType::kINT32;
		condition &= inOut[pos].type == inOut[0].type;
		return condition;
	}

	const char* getPluginType() const override
	{
		return "ReluPluginV2IO";
	}

	const char* getPluginVersion() const override
	{
		return "2";
	}

	void destroy() override
	{
		delete this;
	}

	IPluginV2Ext* clone() const override
	{
		auto* plugin = new ReluPluginV2IO(*this);
		return plugin;
	}

	DataType getOutputDataType(int index, const DataType* inputTypes,
		int nbInputs) const override
	{
		assert(inputTypes && nbInputs == 1);
		(void)index;
		return inputTypes[0];
	}

	void setPluginNamespace(const char* pluginNamespace) override
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mPluginNamespace.c_str();
	}

	bool isOutputBroadcastAcrossBatch(int outputIndex,
		const bool* inputIsBroadcasted, int nbInputs) const override
	{
		return false;
	}

	bool canBroadcastInputAcrossBatch(int inputIndex) const override
	{
		return false;
	}

};