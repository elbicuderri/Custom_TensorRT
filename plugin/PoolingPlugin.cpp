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

#include "PoolingPlugin.h"

PoolingPluginV2IO::PoolingPluginV2IO(ITensor* data, int BatchSize, int ic, int ih, int iw,
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

PoolingPluginV2IO::PoolingPluginV2IO(ITensor& data, int BatchSize, int ic, int ih, int iw,
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

PoolingPluginV2IO::~PoolingPluginV2IO() override
{
    std::cout << "Pooling plugin disapper~" << std::endl;
}

int PoolingPluginV2IO::getNbOutputs() const override { return 1; }

int PoolingPluginV2IO::initialize() override { return 0; }

void PoolingPluginV2IO::terminate() override {}

Dims PoolingPluginV2IO::getOutputDimensions(int index,
    const Dims* inputs, int npInputDims) override
{
    return Dims{ 3, {iC, oH, oW} };
}

size_t PoolingPluginV2IO::getWorkspaceSize(int maxBatchSize) const override { return 0; }

/*
이 함수에서 cuda kernel이 실행된다.
가장 중요한 포인트는 iType을 어떻게 config나 input dtype에 따라 바꿀수 있느냐이다.
그 방법은 configurePlugin 함수에 있다.
*/
int PoolingPluginV2IO::enqueue(int batchSize, const void* const* inputs,
    void** outputs, void* workspace, cudaStream_t stream) override
{

    if (iType == DataType::kINT8)
    {
        maxpooling_plugin_int8((char*)outputs[0], (char*)inputs[0],
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
        maxpool_plugin((float*)outputs[0], (float*)inputs[0],
            N, iC, iH, iW, kH, kW, pH, pW, sH, sW, stream);
    }

    return 0;
}

/*
밑의 두 함수는 Engine 파일을 저장할 때 사용하는 함수인 거 같은데... 사용법은 잘 모르겠다.
*/
size_t PoolingPluginV2IO::getSerializationSize() const override
{
    return 0;
}

void PoolingPluginV2IO::serialize(void* buffer) const override
{
}

/*
iType = in[0].type;
이 줄에서 in 이 이전 layer의 output data임을 확인했다.
그리고 이 함수가 enqueue 이 전에 실행되기에 iType이 우리가 원하는 dtype으로 바뀐다.
아마도 이 방식을 이용하면 TensorFormat도 쉽게 가능할 거 같기도 하다.. 지금 보니..
*/
void PoolingPluginV2IO::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
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

bool PoolingPluginV2IO::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
    int32_t nbInputs, int32_t nbOutputs) const override
{
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= inOut[pos].type != DataType::kINT32;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}

const char* PoolingPluginV2IO::getPluginType() const override
{
    return "PoolingPluginV2IO";
}

const char* PoolingPluginV2IO::getPluginVersion() const override
{
    return "2";
}

void PoolingPluginV2IO::destroy() override
{
    delete this;
}

IPluginV2Ext* PoolingPluginV2IO::clone() const override
{
    auto* plugin = new PoolingPluginV2IO(*this);
    return plugin;
}

DataType PoolingPluginV2IO::getOutputDataType(int index, const DataType* inputTypes,
    int nbInputs) const override
{
    assert(inputTypes && nbInputs == 1);
    (void)index;
    return inputTypes[0];
}

void PoolingPluginV2IO::setPluginNamespace(const char* pluginNamespace) override
{
    mPluginNamespace = pluginNamespace;
}

const char* PoolingPluginV2IO::getPluginNamespace() const override
{
    return mPluginNamespace.c_str();
}

bool PoolingPluginV2IO::isOutputBroadcastAcrossBatch(int outputIndex,
    const bool* inputIsBroadcasted, int nbInputs) const override
{
    return false;
}

bool PoolingPluginV2IO::canBroadcastInputAcrossBatch(int inputIndex) const override
{
    return false;
}