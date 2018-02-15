import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from capsnet import CapsNet


def margin_loss(class_probs, labels):
    correct = F.relu(0.9 - class_probs) ** 2
    wrong = F.relu(class_probs - 0.1) ** 2
    loss = (labels * correct) + 0.5 * ((1 - labels) * wrong)
    loss = loss.sum(dim=-1)
    return loss


def reconstruction_loss(recons, images):
    loss = (recons - images.view(-1, 784)) ** 2
    loss = loss.sum(dim=-1)
    return loss


net = CapsNet()
print net
print "# parameters: ", sum(param.numel() for param in net.parameters())


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

criterion = lambda c, l, r, i: (margin_loss(c, l) + 0.0005 * reconstruction_loss(r, i)).mean()
optimizer = Adam(net.parameters())

for epoch in range(100):
    for i, data in enumerate(trainloader, 0):
        inputs, labs = data
        labels = torch.eye(10).index_select(dim=0, index=labs)
        inputs, labels = Variable(inputs), Variable(labels)
        class_probs, recons = net(inputs)
        acc = torch.mean((Variable(labs) == torch.max(class_probs, dim=-1)[1]).double())
        loss = criterion(class_probs, labels, recons, inputs)
        loss.backward()
        optimizer.step()
        print loss.data[0], acc.data[0]
