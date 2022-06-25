from functools import reduce

import cv2
import torch
from kornia import image_to_tensor, create_meshgrid, tensor_to_image
from matplotlib import pyplot as plt
from torch.nn import functional

from RandomAdjust import RandomAdjust


def test_complex():
    # read sample images
    x = image_to_tensor(cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)).float() / 255
    x.unsqueeze_(0)  # [1, 1, h, w]

    # initialize transform
    # transforms: 'ep' - elastic + perspective ｜ 'e' - elastic only | 'p' - perspective only
    ra = RandomAdjust({'transforms': 'ep', 'kernel_size': (103, 103), 'sigma': (32, 32), 'distortion_scale': 0.3})

    # x -> warped x
    x_w, params = ra(x)

    # inverse grid -> raw grid - dp - de
    h, w = x.size()[-2:]
    disp = reduce(lambda i, j: i + j, [v for _, v in params.items()])
    grid = create_meshgrid(h, w, device=x.device).to(x.dtype)

    # warped x -> reduction x
    x_r = functional.grid_sample(x_w, (grid - disp))

    # merge and display
    x_s = torch.hstack([x_w.squeeze(), x_r.squeeze()])
    plt.imshow(tensor_to_image(x_s * 255), cmap='gray')
    plt.show()


if __name__ == '__main__':
    test_complex()
