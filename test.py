# coding:utf-8
import os
import argparse
from utils import *
import torch
from pathlib import Path
import sys
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from torch.utils.data import DataLoader
# from data.TaskFusion_dataset import Fusion_dataset
from data.dataset import Fusion_dataset
from fusion_net.FusionNet import  RailGuardv1, RailGuardv3
from tqdm import tqdm
from utils.utils import *

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(ir_dir='./test_imgs/ir', vi_dir='./test_imgs/vi', save_dir='./SeAFusion', fusion_model_path='./model/Fusion/fusionmodel_final.pth'):
    # fusionmodel = FusionNet(output=1)
    os.makedirs(save_dir, exist_ok=True)
    fusionmodel = RailGuardv3(in_channels=2, dims= [32, 64, 32, 2])
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path, weights_only=True))
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset(split="val", data_name='MSRS', shape= (480, 640))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    total_time = 0
    count = 0
    with torch.no_grad():
        for it, (img_vis, img_ir, _, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            tic = time.time()
            fused_img = fusionmodel(vi_Y, img_ir)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            toc = time.time()
            duration = (toc - tic) * 1000
            total_time += duration
            count +=1
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))
        mean_time = total_time/count
        print(f"Average run time {mean_time: .3f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusionmodel_final.pth')
    ## dataset
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='./test_imgs/ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='./test_imgs/vi')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)
