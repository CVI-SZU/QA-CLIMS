import torch
import torch.nn as nn


class MaskAdapter_DynamicThreshold(nn.Module):
    def __init__(self, alpha, mask_cam=False):
        super(MaskAdapter_DynamicThreshold, self).__init__()

        self.alpha = alpha
        self.mask_cam = mask_cam

        print(f"MaskAdapter_DynamicThreshold:")
        print(f"  alpha: {alpha}")
        print(f"  mask_cam: {mask_cam}")

    def forward(self, x):
        """

        :param x: cam of each class in a batch, shape: [batch_size * num_classes, 1, h, w]
        :return: masked cam, shape: [batch_size * num_classes, 1, h, w]
        """
        binary_mask = []
        for i in range(x.shape[0]):
            th = torch.max(x[i]) * self.alpha
            binary_mask.append(
                torch.where(x[i] >= th, torch.ones_like(x[0]), torch.zeros_like(x[0]))
            )
        binary_mask = torch.stack(binary_mask, dim=0)

        if self.mask_cam:
            return x * binary_mask
        else:
            return binary_mask

