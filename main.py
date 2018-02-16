from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
import time

from losses import margin_loss, reconstruction_loss
from models.capsnet import CapsNet

if __name__ == '__main__':
    CUDA = torch.cuda.is_available()
    net = CapsNet(1, 6 * 6 *32)
    if CUDA:
        net.cuda()
    print (net)
    print ("# parameters: ", sum(param.numel() for param in net.parameters()))

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./fashion', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./fashion', train=False,
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
            labels_one_hot = torch.eye(10).index_select(dim=0, index=labels)
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
            if CUDA:
                inputs, labels_one_hot, labels = inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
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
            inputs, labels_one_hot, labels = Variable(inputs), Variable(labels_one_hot), Variable(labels)
            if CUDA:
                inputs, labels_one_hot, labels = inputs.cuda(), labels_one_hot.cuda(), labels.cuda()
            class_probs, recons = net(inputs)
            acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
            test_acc += acc.data[0]
        print('[epoch {}/{} done in {:.2f}s] train_acc: {:.5f} test_acc: {:.5f}'.format(epoch + 1, n_epochs, (time.time() - time_start), train_acc/(i + 1), test_acc/(j + 1)))