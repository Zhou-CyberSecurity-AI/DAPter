import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Code.cifar10_vgg import VGG
from Code.dprivter import _Dprivter_cifar10
from Code.get_dataset import loading
import pandas as pd
import numpy as np
from Code.get_dataset import transform_invert, transform


def loss_function(output1, output2, labels):
    lemda = 0
    # Loss_acc
    criterion = nn.CrossEntropyLoss()
    # Loss_n
    ln = nn.L1Loss()
    I_obejct = torch.ones((3, 32, 32)) * 128 / 255.
    I_obejct = I_obejct.cuda()
    convert_loss = ln(output1, I_obejct)
    loss = lemda * convert_loss + (1 - lemda) * criterion(output2, labels)
    return loss, convert_loss


def to_csv(loss, acc):
    loss = pd.DataFrame(loss)
    acc = pd.DataFrame(acc)
    loss.to_csv('../Result/dprivter_dlass_cifar10_vgg16_loss_lamda=0.csv')
    acc.to_csv('../Result/dprivter_dlass_cifar10_vgg16_acc_lamda=0.csv')


def train(**kwargs):
    trainloader = kwargs['dataloader']
    dlass = kwargs['dlass']
    dprivter = kwargs['dprivter']
    Loss = []
    Acc = []
    optimizer = optim.Adam(dprivter.parameters(), lr=0.001)
    for epoach in range(100):
        running_loss = 0.0
        running_acc = 0.0
        entropy_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs_1 = dprivter(inputs)
            # 3, 32, 32 0-1
            outputs_2 = dlass(outputs_1)

            loss, convert_loss = loss_function(outputs_1, outputs_2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            entropy_loss += convert_loss
            _, predicted = torch.max(outputs_2.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / len(inputs)
            running_acc += acc
            if i % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain batch: {}\tLoss: {:.6f}, Entropy loss: {:.6f}, '
                      'Acc: {:.4f}'.format(
                    epoach + 1, i * len(inputs), len(trainloader.dataset),
                    100. * i / len(trainloader), i,
                    loss.item(), convert_loss, 100 * correct / len(inputs)))

        print('====> Epoch: {} Average batch loss: {:.5f}, Average batch entropy loss: {:.5f}, Average batch acc: '
              '{:.5f}'.format(
            epoach + 1, running_loss / len(trainloader), entropy_loss / len(trainloader),
            100 * running_acc / len(trainloader)))
        Loss.append(running_loss / len(trainloader))
        Acc.append(running_acc / len(trainloader))
    to_csv(np.array(Loss), np.array(Acc))
    torch.save(dprivter, '../Result/dprivter_cifar10_lamda=0')
    print('Finished Training')


if __name__ == '__main__':
    # loading dataset
    trainloader, testloader = loading()
    # loading dlass model
    dlass = VGG('VGG11').cuda()
    dlass = torch.load('../Result/cifar10_vgg16')

    # loading dprivter model
    dprivter = _Dprivter_cifar10().cuda()

    train(dataloader=trainloader, dlass=dlass, dprivter=dprivter)
