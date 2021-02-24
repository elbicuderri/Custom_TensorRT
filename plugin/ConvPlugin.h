/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

class ConvPluginV2 : public nvinfer1::IPluginV2Ext
{
/*
class에서 쓰일 멤버 변수 정의.
*/
public:
	DataType iType;
	int N;
	int iC, iH, iW;
	int oC, oH, oW;
	int kH, kW;
	std::vector<float> h_weight;
	std::vector<float> h_bias;
	const char* mPluginNamespace;
	std::string mNamespace;

public:
	ConvPluginV2() = delete;

	ConvPluginV2(ITensor* data, int BatchSize, int oC, int iC, int iH, int iW, 
		int kH, int kW, int pH, int pW, int sH, int sW,
		Weights Weight, Weights Bias);

	~ConvPluginV2() override;

	int getNbOutputs() const override;

	int initialize() override;

	void terminate() override;

	Dims getOutputDimensions(int index,
		const Dims* inputs, int npInputDims) override;

	size_t getWorkspaceSize(int maxBatchSize) const override;

	int enqueue(int batchSize, const void* const* inputs,
		void** outputs, void* workspace, cudaStream_t stream) override;

	size_t getSerializationSize() const override;

	void serialize(void* buffer) const override;

	void configurePlugin(const Dims* inputDims, int nbInputs,
		const Dims* outputDims, int nbOutputs,
		const DataType* inputTypes, const DataType* outputTypes,
		const bool* inputIsBroadcast, const bool* outputIsBroadcast,
		PluginFormat floatFormat, int maxBatchSize) override;

	bool supportsFormat(DataType type, PluginFormat format) const override;

	const char* getPluginType() const override;

	const char* getPluginVersion() const override;

	void destroy() override;

	IPluginV2Ext* clone() const override;

	DataType getOutputDataType(int index, const DataType* inputTypes,
		int nbInputs) const override;

	void setPluginNamespace(const char* pluginNamespace) override;

	const char* getPluginNamespace() const override;

	bool isOutputBroadcastAcrossBatch(int outputIndex,
		const bool* inputIsBroadcasted, int nbInputs) const override;

	bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    
};