#pragma once

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