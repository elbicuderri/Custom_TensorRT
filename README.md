# Custom_TensorRT

## This is a code making custom convolution layer.(using CUDA kernel, TensorRT IPlugin)

> MNIST
>
> Conv2D ( 5*5 kernel, zero padding, OutChannel = 5) : (1, 28, 28) -> (5, 24, 24)
> Maxpooling ( 2*2 kernel, zero padding, 2*2 stride) : (5, 24, 24) -> (5, 12, 12)
> FClayer1 : (5, 12, 12) -> (120,)
> Relu1 : (120,) -> (120,)
> FClayer2 : (120,) -> (10,)
> Softmax : (10,) -> (10,)

## Using TenosoRT API for constructing CNN network.

## INT8 Calibrator, Plugin layer IMPLEMENTATION repo
