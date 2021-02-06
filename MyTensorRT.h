class NetworkRT {

public:
    nvinfer1::DataType dtRT;
    nvinfer1::IBuilder *builderRT;
    nvinfer1::IRuntime *runtimeRT;
    nvinfer1::INetworkDefinition *networkRT; 

#if NV_TENSORRT_MAJOR >= 6  
    nvinfer1::IBuilderConfig *configRT;

#endif
    nvinfer1::ICudaEngine *engineRT;
    nvinfer1::IExecutionContext *contextRT;

    const static int MAX_BUFFERS_RT = 10;
    void* buffersRT[MAX_BUFFERS_RT];
    dataDim_t buffersDIM[MAX_BUFFERS_RT];
    int buf_input_idx, buf_output_idx;

    dataDim_t input_dim, output_dim;
    dnnType *output;
    cudaStream_t stream;

    PluginFactory *pluginFactory;

    NetworkRT(Network *net, const char *name);
    virtual ~NetworkRT();

    int getMaxBatchSize() {
        if(engineRT != nullptr)
            return engineRT->getMaxBatchSize();
        else
            return 0;
    }

    int getBuffersN() {
        if(engineRT != nullptr)
            return engineRT->getNbBindings();
        else 
            return 0;
    }

    /**
        Do inference
    */
    dnnType* infer(dataDim_t &dim, dnnType* data);
    void enqueue(int batchSize = 1);    

    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Layer *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Conv2d *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Activation *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Dense *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Pooling *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Softmax *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Route *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Flatten *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Reshape *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Reorg *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Region *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Shortcut *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Yolo *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Upsample *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, DeformConv2d *l);

    bool serialize(const char *filename);
    bool deserialize(const char *filename);
};
