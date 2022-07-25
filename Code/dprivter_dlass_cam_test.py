import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from Code.utils.grad_cam import GradCAM, show_cam_on_image, center_crop_img
from Code.get_dataset import loading, transform_invert
from Code.utils.image import imshow
import torchvision


def cam_result(model, target_layers, input_tensor, img, target_category):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    # 32, 32
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    return visualization


def CAM(testloader):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    model = torch.load('../Result/cifar10_vgg16')
    target_layers = [model.features[27]]
    print(model.features)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataiter = iter(testloader)

    images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    net = torch.load('../Result/dprivter_cifar10_lamda=0.95')
    with torch.no_grad():
        images_test = images.cuda()
        output1 = net(images_test).cpu()
    for i in range(1, 11):
        j = i+10
        img1 = transform_invert(images[j], data_transform).convert('RGB')
        plt.subplot(4, 10, i)
        plt.imshow(img1)
        plt.axis('off')

        plt.subplot(4, 10, i + 10)
        img2 = output1[j, :, :, :]
        img2 = transform_invert(img2, data_transform).convert('RGB')
        plt.imshow(img2)
        plt.axis('off')

        input_test = torch.unsqueeze(images[j], dim=0)
        img1 = transform_invert(images[j], data_transform).convert('RGB')
        img1 = np.array(img1, dtype=np.uint8)
        img3 = cam_result(model, target_layers, input_test, img1, int(labels[i]))

        plt.subplot(4, 10, i + 20)
        plt.imshow(img3)
        plt.axis('off')

        img2 = np.array(img2, dtype=np.uint8)
        img2_test = data_transform(img2)
        input_test = torch.unsqueeze(img2_test, dim=0)
        img4 = cam_result(model, target_layers, input_test, img2, int(labels[i]))
        plt.subplot(4, 10, i + 30)
        plt.imshow(img4)
        plt.axis('off')
    plt.tight_layout(0.1)
    plt.show()


if __name__ == '__main__':
    _, testloader = loading()
    CAM(testloader)
