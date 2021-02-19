"""
This code is forked from https://github.com/elbicuderri/C_cuDNN_CUDA
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, Linear
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

print("Hi!")

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 100
transform = ToTensor()

#image to Tensor -> image, label
train_dataset = MNIST('C:\\data\\mnist_data/',
                               download=True,
                               train=True,
                               transform=transform)

test_dataset = MNIST("C:\\data\\mnist_data/",
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

print("data ready")

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv = Conv2d(1, 5, kernel_size=5, padding=2, bias=False)
        self.batchnorm = BatchNorm2d(5, eps=0.001)
        self.maxpool = MaxPool2d(2, stride=2)
        self.dense1 = Linear(5 * 14 * 14, 120)
        self.dense2 = Linear(120, 10)
        self.mode = 0

    def forward(self, x):
        x = x.float()
        conv = self.conv(x)
        batchnorm = self.batchnorm(conv)
        maxpool = self.maxpool(batchnorm)
        relu_maxpool = F.relu(maxpool)
        flatten = relu_maxpool.view(-1, 5 * 14 * 14)
        dense1 = self.dense1(flatten)
        relu_dense1 = F.relu(dense1)
        dense2 = self.dense2(relu_dense1)
        result = F.softmax(dense2, dim=1)
        
        if self.mode == 1:
            def save_value(value, name):
                value_arr = value.cpu().detach().numpy()
                print(name, ": ", value_arr.shape)
                value_arr.tofile(f"../value/{name}_pytorch.bin")

            value_list = [conv, batchnorm, maxpool, relu_maxpool, flatten, dense1, relu_dense1, dense2, result]
            name_list = ["conv", "batchnorm", "maxpool", "relu_maxpool", "flatten", "dense1", "relu_dense1", "dense2", "result"]
            for v, n in zip(value_list, name_list):
                save_value(v, n)

        return result


model = mnist_model()
model.to(device)

print(model)

optimizer = Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    valid_loss = 0
    correct = 0
    for (valid_data, valid_target) in valid_loader:
        with torch.no_grad():
            valid_data, valid_target = valid_data.to(device), valid_target.to(device)
            valid_output = model(valid_data)
            # sum up batch loss
            valid_loss += F.nll_loss(valid_output, valid_target, reduction='sum').item()
            # get the index of the max log-probability
            pred = valid_output.argmax(1, keepdim=True)
            correct += pred.eq(valid_target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    print("")
    print(f"Valid set: Average loss: {valid_loss:.4f}, Accuracy: {correct}/{len(valid_loader.dataset)}\
        ({100. * correct / len(valid_loader.dataset):.2f}%)\n")
          
    # print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(valid_loader.dataset),
    #     100. * correct / len(valid_loader.dataset)))
    
mean_list = []
var_list = []

for epoch in range(0, 2):
    train(epoch)
    mean = model.batchnorm.running_mean.clone()
    # print(f"{epoch}th running_mean: ", mean)
    variance = model.batchnorm.running_var.clone()
    # print(f"{epoch}th running_variance: ", variance)
    test()
    mean_list.append(mean)
    var_list.append(variance)

print(mean_list)
print(var_list)

##==============================================================================

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    model.mode = 1
    model.eval()
    _ = model(data)

def save_weights(name, weight):
    weight = weight.cpu().detach().numpy()
    print(name, ": ", weight.shape)
    weight.tofile(f"../weight/{name}_pytorch.bin")

print("=======================================================================")

name_list = ["kernel", "gamma", "beta", "W1", "b1", "W2", "b2"]

parameters = list(model.parameters())

# for w, n in zip(parameters[:3], name_list[:3]):
#     save_weights(w, n)

for name_weight in list(model.named_parameters())[:3]:
    save_weights(name_weight[0], name_weight[1])

# save_weights(mean_list[-1], "mean")
# save_weights(var_list[-1], "variance")

save_weights("mean", mean_list[-1])
save_weights("variance", var_list[-1])

# for w, n in zip(parameters[3:], name_list[3:]):
#     save_weights(w, n)

for name_weight in list(model.named_parameters())[3:]:
    save_weights(name_weight[0], name_weight[1])

print("Finished!")
