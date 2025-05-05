# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
import os, sys
import numpy as np
from PIL import Image
from torchvision import transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

from fusion_net.experimental import MobileNetV3Block
from fusion_net.building_blocks import *


def tensor_to_image(tensor, output_path):
    """
    Converts a PyTorch tensor to a .png image and saves it.
    
    Args:
        tensor (torch.Tensor): Tensor to be converted (shape: [1, H, W]).
        output_path (str): Path to save the image file.
    """
    # Detach tensor from computation graph, convert to CPU, and numpy
    array = tensor.squeeze(0).detach().cpu().numpy()
    # Normalize to range [0, 255]
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)
    # Save the image
    image = Image.fromarray(array)
    image.save(output_path)

class RailGuard(nn.Module):
    """
    RailGuard, pixel wise adpative fusion model
    ===========================================
    
    in_channels: int
    out_channles: int
    dims: list
    """
    def __init__(self, in_channels = 2, dims= [32, 64, 32, 2]):
        super(RailGuard, self).__init__()
        self.dims = dims
        
        # Pixel wise feature extraction
        self.init_block = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0]),
			nn.ReLU(),
		) 
        self.basic_block = BasicBlock(channel_num=dims[0])
        
        self.d1 =  ConvBNReLU(in_channels=dims[0], out_channels=dims[1], kernel_size=3)
        self.d2 =  ConvBNReLU(in_channels=dims[1], out_channels=dims[2], kernel_size=3)
        self.d3 =  ConvBNReLU(in_channels=dims[2], out_channels=dims[3], kernel_size=1, padding=0)
    
    def normalize(self, x):
        min_val = x.min()
        max_val = x.max()
        return (x - min_val) / (max_val - min_val)

    def forward(self, ir, vi):
        # Concat along channel dims
        x = torch.cat([ir, vi], dim=1)
        
        x = self.init_block(x)
        x = self.basic_block(x)
        
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        
        # Adaptive pixel-wise fusion
        x_ir, x_vi = x[:, 0:1, :, :], x[:, 1:2, :, :]
        
        # Save the intermediate tensors as images
        tensor_to_image(x_ir[0], 'x_ir.png')  # Save x_ir as an image
        tensor_to_image(x_vi[0], 'x_vi.png')  # Save x_vi as an image
        
        ir_new, vi_new = ir * x_ir, vi * x_vi
        fused = ir_new + vi_new
        fused = self.normalize(fused)
        return fused

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,2,480,640).astype(np.float32))
    model = RailGuard(in_channels=2, dims=[16, 32, 64])
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

def load_ir_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0)

def load_vi_image_as_y_channel(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("YCbCr")
    y_channel, _, _ = image.split()
    return transform(y_channel).unsqueeze(0)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vi", type = str)
    parser.add_argument("--ir", type = str)
    opt = parser.parse_args()
    
    ir_tensor = load_ir_image_as_tensor(opt.ir)
    vi_tensor = load_vi_image_as_y_channel(opt.vi)
    
    print(f"Infrared image tensor shape: {ir_tensor.shape}")
    print(f"Visible image Y channel tensor shape: {vi_tensor.shape}")
    # ir = torch.rand(1, 1, 480, 640).to(device)
    # vi = torch.rand(1, 1, 480, 640).to(device)
    
    model = RailGuard(in_channels=2, dims=[32, 64, 32, 2]).to(device)
    out =  model(ir_tensor.cuda(), vi_tensor.cuda())
    # print("Warmup")
    # for _ in range(10):
    #     out = model(ir, vi)
           
    # print("Start inference")
    
    # tic = time.time()
    # for _ in range(100):
    #     out = model(ir, vi)
    # toc = time.time()
    # duration = ((toc - tic) * 1000)/100
    
    # print(f"Time taken: {duration: .4f}")
    # print(out.shape)
    
    
    
