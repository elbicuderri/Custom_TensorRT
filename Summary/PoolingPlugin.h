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

void maxpooling(float *output, float *input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width,
	int pad_height, int pad_width, int stride_height, int stride_width, cudaStream_t stream);

void maxpooling_int8(char*output, char* input,
	int batch, int channel, int height, int width,
	int kernel_height, int kernel_width,
	int pad_height, int pad_width, int stride_height, int stride_width, cudaStream_t stream);

/*
TensorRT에서 CUDA Kernel을 직접 만들어서 Plugin Layer를 생성하는 방법이다.

현재까지 implement한 정도
- activation, pooling 같이 weight가 없는 layer는 어떤 dtype도 plugin은 가능. 하지만, tensorformat이 앞뒤로
다르면 optimization시 reformat layer가 추가됨. 이 부분을 추가하려면 고생을 좀 더 해야됨.

- weight가 있는 layer를 custom plugin해야 될 경우 8bit인 경우에는 calibrator를 써서 자동으로 8bit inference가
이루어지게 하는 방법은 아예 불가능한 것 같음.. 다른 방법으로 layer마다 activation, weight의 scale factor를 직접 구해서
넣어주어야 하는 것 같다.
https://forums.developer.nvidia.com/t/is-there-a-no-way-to-get-quantized-weights-after-calibration/160389/6

특히 dtype을 맞춰줘도 TensorRT에서는 TensorFormat이라는 것도 최적화가 진행되어서 이것을 안 맞춰주면 reformat layer가 생김.
그냥 TensorRT가 quantization을 잘 지원해주길 기다리자... 사용법이나 잘 익히자...

설명이 안 달린 함수는 특별히 건드릴 부분이 없는 함수이다.
*/
class PoolingPluginV2IO : public nvinfer1::IPluginV2IOExt
{
/*
사용하고 싶은 멤버 변수 정의
*/
public:
	DataType iType;
	int N;
	int iC, iH, iW;
	int kH, kW;
	int pH, pW;
	int sH, sW;
	int oH, oW;
	std::string mPluginNamespace;
	std::string mNamespace;

public:
	PoolingPluginV2IO() = delete;

	PoolingPluginV2IO(ITensor* data, int BatchSize, int ic, int ih, int iw,
		int kh, int kw, int ph, int pw, int sh, int sw)
		: N(BatchSize)
		, iC(ic)
		, iH(ih)
		, iW(iw)
		, kH(kh)
		, kW(kw)
		, pH(ph)
		, pW(pw)
		, sH(sh)
		, sW(sw)
	{
		iType = data->getType();
		oH = ((iH - 2 * pH - kH) / sH) + 1;
		oW = ((iW - 2 * pW - kW) / sW) + 1;
	}

	PoolingPluginV2IO(ITensor& data, int BatchSize, int ic, int ih, int iw,
		int kh, int kw, int ph, int pw, int sh, int sw)
		: N(BatchSize)
		, iC(ic)
		, iH(ih)
		, iW(iw)
		, kH(kh)
		, kW(kw)
		, pH(ph)
		, pW(pw)
		, sH(sh)
		, sW(sw)
	{
		iType = data.getType();
		oH = ((iH - 2 * pH - kH) / sH) + 1;
		oW = ((iW - 2 * pW - kW) / sW) + 1;
	}

	~PoolingPluginV2IO() override
	{
		std::cout << "Pooling plugin disapper~" << std::endl;
	}

	int getNbOutputs() const override { return 1; }

	int initialize() override { return 0; }

	void terminate() override {}

	Dims getOutputDimensions(int index,
		const Dims* inputs, int npInputDims) override
	{
		return Dims{ 3, {iC, oH, oW} };
	}

	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }


	/*
	이 함수에서 cuda kernel이 실행된다.
	가장 중요한 포인트는 iType을 어떻게 config나 input dtype에 따라 바꿀수 있느냐이다.
	그 방법은 configurePlugin 함수에 있다.
	*/
	int enqueue(int batchSize, const void* const* inputs,
		void** outputs, void* workspace, cudaStream_t stream) override
	{

		if (iType == DataType::kINT8)
		{
			maxpooling_int8((char*)outputs[0], (char*)inputs[0],
				N, iC, iH, iW, kH, kW, pH, pW, sH, sW, stream);
		}

		/*
		귀찮아서 16비트 cuda kernel은 따로 안 만들었다.
		template cudat kernel을 만드는 방법이 생각보다 어렵다..
		그냥 cu 파일을 따로 만들었다..
		*/

		//else if ((iType == DataType::kHALF)
		//{
		//	maxpooling_half((half_float::half*)outputs[0], (half_float::half*)inputs[0],
		//		N, iC, iH, iW, kH, kW, pH, pW, sH, sW, stream);
		//}

		else
		{
			maxpooling((float*)outputs[0], (float*)inputs[0],
				N, iC, iH, iW, kH, kW, pH, pW, sH, sW, stream);
		}

		return 0;
	}

	/*
	밑의 두 함수는 Engine 파일을 저장할 때 사용하는 함수인 거 같은데... 사용법은 잘 모르겠다.
	*/
	size_t getSerializationSize() const override
	{
		return 0;
	}

	void serialize(void* buffer) const override
	{
	}

	/*
	iType = in[0].type;
	이 줄에서 in 이 이전 layer의 output data임을 확인했다.
	그리고 이 함수가 enqueue 이 전에 실행되기에 iType이 우리가 원하는 dtype으로 바뀐다.
	아마도 이 방식을 이용하면 TensorFormat도 쉽게 가능할 거 같기도 하다.. 지금 보니..
	*/
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
		//assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
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
		return "PoolingPluginV2IO";
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
		auto* plugin = new PoolingPluginV2IO(*this);
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
