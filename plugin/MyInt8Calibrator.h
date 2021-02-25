#pragma once

class MyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{

public:
	int m_Total;
	int m_Batch;
	int m_InputC;
	int m_InputH;
	int m_InputW;
	int m_TotalSize;
	int m_BatchSize;
	std::string m_InputBlobName;
	std::string m_CalibTableFilePath;
	int m_ImageIndex;
	bool m_ReadCache;
	void* m_DeviceInput;
	float* m_Data;
	std::vector<float> h_batchdata;
	std::vector<char> m_CalibrationCache;

	MyInt8Calibrator() = delete; // 혹시나 변수 없는 객체가 생기면 지우겠다는 constructor다. 없어도 된다.

	MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
		const std::string& InputBlobName, const std::string& CalibTableFilePath,
		float* Data, bool ReadCache);

	//MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
	//	const std::string& InputBlobName, const std::string& CalibTableFilePath,
	//	const std::vector<float>& Data, bool ReadCache);

	~MyInt8Calibrator();

	int getBatchSize() const override;

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

	const void* readCalibrationCache(size_t& length) override;

	void writeCalibrationCache(const void* cache, size_t length) override;

};