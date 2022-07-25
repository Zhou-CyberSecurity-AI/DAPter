import torch.nn as nn
import torch
from Code.cifar10_vgg import VGG
from Code.get_dataset import loading


def _test_dlass(test_loader):
    net = VGG('VGG16').cuda()
    net = torch.load('../Result/cifar10_vgg16')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
    _, testloader = loading()
    _test_dlass(testloader)
