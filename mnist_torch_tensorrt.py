import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dtype = "float32"

batch_size = 100

transform = transforms.ToTensor()# 0 ~ 1

train_dataset = datasets.MNIST('C:\\data\\mnist_data/',
                               download=True,
                               train=True,
                               transform=transform) # image to Tensor # image, label

test_dataset = datasets.MNIST("C:\\data\\mnist_data/",
                              train=False,
                              download=True,
                              transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 5, kernel_size=5, padding=0, bias=True)
        # self.batchnorm = nn.BatchNorm2d(5, eps=0.001)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(5 * 12 * 12, 120)
        self.dense2 = nn.Linear(120, 10)
        self.dump = 0

    def forward(self, x):
        insize = x.size(0)
        x = x.float()
        conv1 = self.conv(x)
        # batchnorm = self.batchnorm(conv1)
        maxpool = self.maxpool(conv1)
        # relu_maxpool = F.relu(maxpool)
        flatten = maxpool.view(insize, -1)
        dense1 = self.dense1(flatten)
        relu_dense1 = F.relu(dense1)
        dense2 = self.dense2(relu_dense1)
        y = F.softmax(dense2, dim=1)

        if self.dump == 1:
            conv1_arr = conv1.cpu().data.numpy().astype(dtype)
            print(conv1_arr.shape)
            conv1_arr.tofile(f"value/conv1_torch_{dtype}.bin")
            # np.save(conv1_arr, f"value/conv1_torch_{dtype}.npy")

            # batchnorm_arr = batchnorm.cpu().data.numpy()
            # print(batchnorm_arr.shape)
            # batchnorm_arr.tofile("value/batchnorm_torch_{dtype}.bin")
            maxpool_arr = maxpool.cpu().data.numpy().astype(dtype)
            print(maxpool_arr.shape)
            maxpool_arr.tofile(f"value/maxpool1_torch_{dtype}.bin")
            # np.save(maxpool_arr, f"value/maxpool1_torch_{dtype}.npy")

            # relu_maxpool_arr = relu_maxpool.cpu().data.numpy()
            # print(relu_maxpool_arr.shape)
            # relu_maxpool_arr.tofile("value/relu_maxpool_torch_{dtype}.bin")
            flatten_arr = flatten.cpu().data.numpy().astype(dtype)
            print(flatten_arr.shape)
            flatten_arr.tofile(f"value/flatten_torch_{dtype}.bin")
            # np.save(flatten_arr, f"value/flatten_torch_{dtype}.npy")

            dense1_arr = dense1.cpu().data.numpy().astype(dtype)
            print(dense1_arr.shape)
            dense1_arr.tofile(f"value/dense1_torch_{dtype}.bin")
            # np.save(dense1_arr, f"value/dense1_torch_{dtype}.npy")

            relu_dense1_arr = relu_dense1.cpu().data.numpy().astype(dtype)
            print(relu_dense1_arr.shape)
            relu_dense1_arr.tofile(f"value/relu_dense1_torch_{dtype}.bin")
            # np.save(relu_dense1_arr, f"value/relu_dense1_torch_{dtype}.npy")

            dense2_arr = dense2.cpu().data.numpy().astype(dtype)
            print(dense2_arr.shape)
            dense2_arr.tofile(f"value/dense2_torch_{dtype}.bin")
            # np.save(dense2_arr, f"value/dense2_torch_{dtype}.npy")

            result = y.cpu().data.numpy().astype(dtype)
            print(result.shape)
            result.tofile(f"value/result_torch_{dtype}.bin")
            # np.save(result, f"weights_torch/result_torch_{dtype}.npy")

        return y

model = Net()
print(model)
model.to(device)
parameters_zero = list(model.parameters())
# print("params_zero: ", parameters_zero)s
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

# mean_list = []
# var_list = []
#
for epoch in range(0, 10):
    train(epoch)
    # mean = model.batchnorm.running_mean.clone()
    # print(f"{epoch}th running_mean: ", mean)
    # variance = model.batchnorm.running_var.clone()
    # print(f"{epoch}th running_variance: ", variance)
    test()
    # mean_list.append(mean)
    # var_list.append(variance)

# print(mean_list)
# print(var_list)

def save_weights(weights, name):
    weights = weights.cpu().detach().numpy().astype(dtype)
    print(name, ": ", weights.shape)
    weights.tofile(f"weights/{name}_torch_{dtype}.wts")
    # np.save(weights, f"weights_torch/{name}_torch_{dtype}.npy")

parameters = list(model.parameters())

name_list = ["conv1filter", "conv1bias", "ip1filter", "ip1bias", "ip2filter", "ip2bias"]

for w, n in zip(parameters, name_list):
    save_weights(w, n)
# =========================================================================================
# ip1filter = parameters[2].cpu().detach().numpy().astype('float32')
# print(ip1filter.shape)
# for h in range(120):
#     for w in range(720):
#         print(ip1filter[h , w])
# for i in range(2 * 120):
#     print(ip1filter[i])
# =========================================================================================

# save_weights(mean_list[-1], "mean")
# save_weights(var_list[-1], "variance")

def calculate():
    # model.dump = 1
    # model.eval()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = Variable(data).to(device), Variable(target).to(device)
    #     output = model(data)
    for data, target in test_loader:
        data, target = Variable(data).to(device), Variable(target).to(device)
    model.dump = 1
    model.eval()
    output = model(data)

calculate()

print("Finished!")
