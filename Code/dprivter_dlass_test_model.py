import torch
from Code.get_dataset import loading, transform_invert, transform
from Code.dprivter import _Dprivter_cifar10
import matplotlib.pyplot as plt
import numpy as np
from Code.dlass_test_model import _test_dlass
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader


def _test_dprivter(net, testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    with torch.no_grad():
        images_test = images.cuda()
        output1 = net(images_test).cpu()
    k = 0
    plt.figure(figsize=(10, 10), dpi=300)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    for i in range(10):
        for j in range(1, 11, 2):
            plt.subplot(10, 10, i * 10 + j)
            plt.imshow(transform_invert(images[k], transform))
            plt.axis('off')
            output1_1 = output1[k, :, :, :]
            img = transform_invert(output1_1, transform)
            # img = np.array(img, dtype=np.uint8)
            # img = img[:, :, 0]
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.imshow(img)
            plt.axis('off')
            k += 1
    plt.tight_layout(pad=0.1)
    plt.margins(0, 0)
    plt.show()
    plt.close()


def _test_acc(net, testloader):
    correct_rate = _test_dlass(testloader)
    # print('Accuracy of the network on the test images: %d %%' % (correct_rate))
    count = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            my_dataSet = TensorDataset(outputs, labels)
            dataloader = DataLoader(dataset=my_dataSet, batch_size=64, num_workers=0, shuffle=True)
            correct_rate += _test_dlass(dataloader)
            count += 1
    print('Accuracy of the network on the test images: %.2f %%' % (correct_rate / count))


if __name__ == '__main__':
    net = _Dprivter_cifar10().cuda()
    net = torch.load('../Result/dprivter_cifar10_lamda=0.9')

    _, testloader = loading()
    _test_dprivter(net, testloader)
    # _test_acc(net, testloader)
