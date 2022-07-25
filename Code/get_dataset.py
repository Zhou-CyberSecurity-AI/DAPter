import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageTk, Image
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


def loading():
    trainset = torchvision.datasets.CIFAR10(root='../Dataset/data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../Dataset/data', train=False, download=False, transform=transform)

    trianloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)

    return trianloader, testloader


if __name__ == '__main__':
    trainloader, testloader = loading()
    for i, data in enumerate(trainloader, 0):
        a, b = data
        print(a.shape[0])
        exit(0)
