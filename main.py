from __future__ import print_function

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.cuda
import time
from torch.autograd import Variable
from torch.optim import Adam
from models.capsnet import CapsNet


def margin_loss(class_probs, labels):
    correct = F.relu(0.9 - class_probs) ** 2
    wrong = F.relu(class_probs - 0.1) ** 2
    loss = (labels * correct) + 0.5 * ((1. - labels) * wrong)
    loss = loss.sum(dim=-1)
    return loss.mean()


def reconstruction_loss(recons, images):
    loss = (recons - images.view(-1, 784)) ** 2
    loss = loss.sum(dim=-1)
    return loss.mean()


CUDA = torch.cuda.is_available()
net = CapsNet()
if CUDA:
    net.cuda()
print (net)
print ("# parameters: ", sum(param.numel() for param in net.parameters()))

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)
optimizer = Adam(net.parameters())

n_epochs = 30
print_every = 200 if CUDA else 2


for epoch in range(n_epochs):
    train_acc = 0.
    time_start = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # print(torch.max(inputs))
        labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
        if CUDA:
            inputs, labels_one_hot, labels = Variable(inputs).cuda(), Variable(labels_one_hot).cuda(), Variable(labels).cuda()
        else:
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
        optimizer.zero_grad()
        class_probs, recons = net(inputs, labels)
        acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
        train_acc += acc.data[0]
        loss = (margin_loss(class_probs, labels_one_hot) + 0.0005 * reconstruction_loss(recons, inputs))
        loss.backward()
        optimizer.step()
        if (i+1) % print_every == 0:
            print('[epoch {}/{}, batch {}] train_loss: {:.5f}, train_acc: {:.5f}'.format(epoch + 1, n_epochs, i + 1, loss.data[0], acc.data[0]))
    test_acc = 0.
    for j, data in enumerate(testloader, 0):
        inputs, labels = data
        labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
        if CUDA:
            inputs, labels_one_hot, labels = Variable(inputs).cuda(), Variable(labels_one_hot).cuda(), Variable(labels).cuda()
        else:
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
        class_probs, recons = net(inputs)
        acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
        test_acc += acc.data[0]
    print('[epoch {}/{} done in {:.5f}s] train_acc: {:.5f} test_acc: {:.5f}'.format(epoch + 1, n_epochs, (time.time() - time_start), train_acc/(i + 1), test_acc/(j + 1)))