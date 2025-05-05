#!/usr/bin/python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021 Linfeng Tang
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from PIL import Image
import numpy as np
from torch.autograd import Variable
from fusion_net.FusionNet import RailGuardv1, RailGuardv3
from data.TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger.logger import setup_logger
from segmentation.model_TII import BiSeNet
from cityscapes.cityscapes import CityScapes
from losses.loss import OhemCELoss, Fusionloss
from optimizers.optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train_seg(i=0, logger=None, args=None):
    load_path = f'{args.save_path}/fusion/joint_model_final.pth'
    modelpth = f'{args.save_path}/fusion'
    Method = 'Fusion'
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    # dataset
    n_classes = 9
    n_img_per_gpu = args.batch_size
    n_workers = 4
    cropsize = [640, 480]
    ds = CityScapes('./MSRS/', cropsize=cropsize, mode='train', Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i>0:
        net.load_state_dict(torch.load(load_path))
    net.cuda()
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i*20000
    iter_nums=args.seg_epochs

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    # train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = lossp + 0.75 * loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int(( max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start+it + 1, max_it= max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
    # dump the final model
    save_pth = osp.join(modelpth, 'joint_model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')

def train_fusion(num=0, logger=None, args=None, best_loss=None, best_avg_epoch_loss=None):
    # num: control the segmodel 
    lr_start = 0.001
    modelpth = f'{args.save_path}/fusion'
    os.makedirs(modelpth, exist_ok=True)
    fusionmodel = RailGuardv3(in_channels=2, dims= [32, 64, 32, 2])
    fusionmodel.cuda()
    if args.pretrained or num > 0:
        fusion_ckpt = osp.join(modelpth, "last_fusion_model.pth")
        fusionmodel.load_state_dict(torch.load(fusion_ckpt))
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    if num>0:
        n_classes = 9
        segmodel = BiSeNet(n_classes=n_classes)
        save_pth = osp.join(modelpth, 'joint_model_final.pth')
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        segmodel.load_state_dict(torch.load(save_pth))
        segmodel.cuda()
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    
    train_dataset = Fusion_dataset(split='train', 
                                   dataset_root=args.dataset_root)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    if num>0:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 640 * 480 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = Fusionloss()
    
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    avg_epoch_loss = 0
    for epo in range(0, args.fusion_epochs):
        # print('\n| epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
            # print(len(train_loader))
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            label = Variable(label).cuda()
            logits = fusionmodel(image_vis_ycrcb[:, 0:1, :, :], image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)
            optimizer.zero_grad()
            # seg loss
            if num>0:
                out, mid = segmodel(fusion_image)
                lossp = criteria_p(out, lb)
                loss2 = criteria_16(mid, lb)
                seg_loss = lossp + 0.1 * loss2
            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, label, logits,num
            )
            if num>0:
                loss_total = loss_fusion + (num) * seg_loss
            else:
                loss_total = loss_fusion
            avg_epoch_loss += loss_total
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * args.fusion_epochs - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                if num>0:
                    loss_seg=seg_loss.item()
                else:
                    loss_seg=0
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * args.fusion_epochs,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
            # Save best and last checkpoints 
            if loss_total < best_loss:
                best_loss = loss_total
                fusion_model_file = os.path.join(modelpth, 'best_fusion_model.pth')
                torch.save(fusionmodel.state_dict(), fusion_model_file)
           
            fusion_model_file = os.path.join(modelpth, 'last_fusion_model.pth')
            torch.save(fusionmodel.state_dict(), fusion_model_file)
                
        avg_epoch_loss = avg_epoch_loss/ train_loader.n_iter  
        if avg_epoch_loss < best_avg_epoch_loss:
            best_avg_epoch_loss = avg_epoch_loss  
            fusion_model_file = os.path.join(modelpth, 'epoch_best_fusion_model.pth')
            torch.save(fusionmodel.state_dict(), fusion_model_file)
            logger.info("Epoch best fusion Model Save to: {}".format(fusion_model_file))
            logger.info('\n')
        avg_epoch_loss = 0
        
    return best_loss, best_avg_epoch_loss
    

def run_fusion(type='train', args=None):
    fusion_model_path = f"{args.save_path}/fusion/best_fusion_model.pth"
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = RailGuardv3(in_channels=2, dims= [32, 64, 32, 2])
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(dataset_root=args.dataset_root, split=type, type="MSRS")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb[:, 0:1, :, :], images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                 :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='RailGuard')
    parser.add_argument("--dataset_root", default="/drive/faizanai.rrl/faizan/APWNet/datasets/MSRS/msrs/segmentation", type = str, help="dataset root")
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument("--fusion_epochs", type=int, default=10)
    parser.add_argument("--seg_epochs", type=int, default=20000)
    parser.add_argument("--pretrained", action= "store_true")
    parser.add_argument("--M", type = int, default=4, help="maximum epochs for joint training")
    parser.add_argument("--save_path", type = str, default= "runs/", help= "path to save logs, results, weights")
    args = parser.parse_args()
    
    # set up paths
    runs_path = args.save_path
    logpath=f'{runs_path}/logs'
    os.makedirs(runs_path, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)
    
    # Logger
    logger = logging.getLogger()
    setup_logger(logpath)
    b_loss, avg_epoch_loss = float(np.inf), float(np.inf)
    # Joint Training
    for i in range(args.M):
        best_loss, best_avg_epoch_loss = train_fusion(i, logger, args, b_loss, avg_epoch_loss)  
        b_loss = best_loss
        avg_epoch_loss = best_avg_epoch_loss
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion('train', args=args)  
        print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        train_seg(i, logger, args)
        print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")
