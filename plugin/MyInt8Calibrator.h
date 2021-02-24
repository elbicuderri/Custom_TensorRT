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

class MyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
/*
이 class 안에서 내가 사용하고 싶은 variable들을 정의한다.
숫자나 string 경우에는 const를 사용해도 좋으나 array나 vector에 const를 붙일 경우(멤버 변수에)
error가 자주 난다. (이유를 설명하기 좀 복잡하다...) 그러니까 숫자랑 string정도에만 const를 붙이자.(붙이고 싶으면)
*/
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
	void* m_DeviceInput{ nullptr };
	std::vector<char> m_CalibrationCache;
	float* m_Data{ nullptr };
	std::vector<float> m_data;
	std::vector<float> h_batchdata;

	MyInt8Calibrator() = delete; // 혹시나 변수 없는 객체가 생기면 지우겠다는 constructor다. 없어도 된다.

	/*
	Data가 float* 형태로 들어왔을 때의 constructor(생성자)
	*/
	MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
		const std::string& InputBlobName, const std::string& CalibTableFilePath,
		float* Data, bool ReadCache = true);


	/*
	Data가 std::vector<float> 형태로 들어왔을 때의 constructor(생성자)
	*/
	MyInt8Calibrator(int Total, int Batch, int InputC, int InputH, int InputW,
		const std::string& InputBlobName, const std::string& CalibTableFilePath,
		const std::vector<float>& Data, bool ReadCache = true);

	~MyInt8Calibrator();

	/*
	calibration의 batch size를 확인하는 함수다. 대부분의 경우 1이라고 생각하면 될 거 같다.
	키워드에 짧게 설명해보자면,
	const: 이 멤버 함수 내에서 멤버 변수는 const하다. 어떠한 경우에도 변하지 않는다.
	override: 이 멤버 함수는 부모 클래스에서 상속받은 함수이고 그 함수를 override한다.
	*/
	int getBatchSize() const override;

	/*
	가장 중요한 함수이다. 
	batch data를 gpu에 넘겨주는 함수이다.
	bindings: 우리가 원하는 data(calibration 하고 싶은 batch data)를 넘겨주고 싶은 gpu pointer.
	names: network input layer의 name
	nbBindings: 현재는 쓰이지 않았다. 아마 input이 여러개인 경우 필요할 것으로 예측된다..
				bindings[nbBindings] = m_DeviceInput[nbBindings] // 이런 식으로 쓰이지 않을까?
	*/
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;


	/*
	밑의 두 함수는 어딘가에서 그대로 복사해서 돌리니 잘 돌아가고 있으니, 일단은 건드리지 말자.
	calibrationtable 파일이 있으면 있어서 8bit inference를 하고 없으면 table파일을 작성해주는
	함수다.
	*/
	const void* readCalibrationCache(size_t& length) override;

	void writeCalibrationCache(const void* cache, size_t length) override;

};