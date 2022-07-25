from torchsummary import summary
import torch.nn as nn
import torch


class _Dprivter_cifar10(nn.Module):
    def __init__(self):
        super(_Dprivter_cifar10, self).__init__()
        self.Conv_first = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.Conv_second = nn.Sequential(
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, stride=2, kernel_size=2)
        )
        self.Conv_third = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=16, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=3, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.skip(self.Conv_first(x))
        x2 = self.Conv_second(self.Conv_first(x))
        x = torch.add(x1, x2)
        o = self.Conv_third(x)
        return o


if __name__ == '__main__':
    model = _Dprivter_cifar10().cuda()
    summary(model, (3, 224, 224))
