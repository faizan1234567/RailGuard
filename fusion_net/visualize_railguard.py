# coding:utf-8
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import sys
from PIL import Image
from torchvision import transforms


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from fusion_net.building_blocks import *

def tensor_to_image_with_heatmap(tensor, output_path, heatmap=True, original_image=None):
    """
    Converts a PyTorch tensor to an image with an optional heatmap overlay.

    Args:
        tensor (torch.Tensor): Tensor to be converted (shape: [1, H, W]).
        output_path (str): Path to save the image file.
        heatmap (bool): Whether to overlay a heatmap.
        original_image (torch.Tensor or PIL.Image): Original image for overlay.
    """
    # Detach tensor, convert to numpy, and normalize
    array = tensor.squeeze(0).detach().cpu().numpy()
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)

    if heatmap:
        # Create a heatmap
        heatmap_array = np.uint8(255 * (array / 255))
        heatmap_image = Image.fromarray(heatmap_array).convert("L")
        
        # Resize to match the original image size if provided
        if original_image is not None:
            if isinstance(original_image, torch.Tensor):
                original_image = original_image.squeeze(0).detach().cpu().numpy()
                original_image = (original_image - original_image.min()) / (
                    original_image.max() - original_image.min()
                ) * 255
                original_image = Image.fromarray(original_image.astype(np.uint8))
            heatmap_image = heatmap_image.resize(original_image.size, resample=Image.BILINEAR)

        heatmap_image = heatmap_image.convert("RGB")
        heatmap_image.save(output_path)
    else:
        # Save as a grayscale image
        image = Image.fromarray(array)
        image.save(output_path)


class RailGuard(nn.Module):
    def __init__(self, in_channels=2, dims=[32, 64, 32, 2]):
        super(RailGuard, self).__init__()
        self.dims = dims

        # Pixel-wise feature extraction
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
        )
        self.basic_block = BasicBlock(channel_num=dims[0])

        self.d1 = ConvBNReLU(in_channels=dims[0], out_channels=dims[1], kernel_size=3)
        self.d2 = ConvBNReLU(in_channels=dims[1], out_channels=dims[2], kernel_size=3)
        self.d3 = ConvBNReLU(in_channels=dims[2], out_channels=dims[3], kernel_size=1, padding=0)

    def normalize(self, x):
        min_val = x.min()
        max_val = x.max()
        return (x - min_val) / (max_val - min_val)

    def forward(self, ir, vi):
        # Concatenate along channel dimensions
        x = torch.cat([ir, vi], dim=1)

        x = self.init_block(x)
        x = self.basic_block(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        # Adaptive pixel-wise fusion
        x_ir, x_vi = x[:, 0:1, :, :], x[:, 1:2, :, :]

        # Save the intermediate tensors as images with heatmaps
        tensor_to_image_with_heatmap(x_ir[0], "x_ir_heatmap.png", heatmap=True, original_image=vi[0])
        tensor_to_image_with_heatmap(x_vi[0], "x_vi_heatmap.png", heatmap=True, original_image=vi[0])

        ir_new, vi_new = ir * x_ir, vi * x_vi
        fused = ir_new + vi_new
        fused = self.normalize(fused)
        return fused


def load_ir_image_as_tensor(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0)


def load_vi_image_as_y_channel(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("YCbCr")
    y_channel, _, _ = image.split()
    return transform(y_channel).unsqueeze(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vi", type=str)
    parser.add_argument("--ir", type=str)
    opt = parser.parse_args()

    ir_tensor = load_ir_image_as_tensor(opt.ir).to(device)
    vi_tensor = load_vi_image_as_y_channel(opt.vi).to(device)

    print(f"Infrared image tensor shape: {ir_tensor.shape}")
    print(f"Visible image Y channel tensor shape: {vi_tensor.shape}")

    model = RailGuard(in_channels=2, dims=[32, 64, 32, 2]).to(device)
    out = model(ir_tensor, vi_tensor)
