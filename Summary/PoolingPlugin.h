#pragma once

class PoolingPluginV2IO : public nvinfer1::IPluginV2IOExt
{
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
		int kh, int kw, int ph, int pw, int sh, int sw);

	PoolingPluginV2IO(ITensor& data, int BatchSize, int ic, int ih, int iw,
		int kh, int kw, int ph, int pw, int sh, int sw);

	~PoolingPluginV2IO() override;

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

	void configurePlugin(
		const PluginTensorDesc* in,
		int nbInput,
		const PluginTensorDesc* out,
		int nbOutput) override;


	bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
		int32_t nbInputs, int32_t nbOutputs) const override;


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