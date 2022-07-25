import torch.optim as optim
import torch.nn as nn
import torch
from Code.cifar10_vgg import VGG
from Code.get_dataset import loading
from Code.dlass_test_model import test_dlass


def train_model(net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoach in range(50):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            num_correct = (predicted == labels).sum().item()
            acc = num_correct / inputs.shape[0]
            running_acc += acc
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Batch: {}\tLoss: {:.6f}, Acc: {:.6f}'.format(
                    epoach, i * len(inputs), len(trainloader.dataset),
                            100. * i / len(trainloader), i,
                            loss.item() / len(data), acc))

        print('====> Epoch: {} Average loss: {:.5f}, Average acc: {:.5f}'.format(
            epoach, running_loss / len(trainloader), running_acc / len(trainloader)))
    torch.save(net, '../Result/cifar10_vgg16')
    print('Finished Training')


if __name__ == '__main__':
    net = VGG('VGG16').cuda()
    trainloader, testloader = loading()
    train_model(net, trainloader)
    test_dlass(testloader)
