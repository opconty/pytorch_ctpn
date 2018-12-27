#-*- coding:utf-8 -*-
#'''
# Created on 18-12-27 上午10:31
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse

import config
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from data import VOCDataset


random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

num_workers = 8
epochs = 20
lr = 1e-3
resume_epoch = 0
pre_weights = os.path.join(config.checkpoints_dir, 'ctpn_keras_weights.pth.tar')


def get_arguments():
    parser = argparse.ArgumentParser(description='Pytorch CTPN For TexT Detection')
    parser.add_argument('--num-workers', type=int, default=num_workers)
    parser.add_argument('--image-dir', type=str, default=config.img_dir)
    parser.add_argument('--labels-dir', type=str, default=config.xml_dir)
    parser.add_argument('--pretrained-weights', type=str,default=pre_weights)
    return parser.parse_args()


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth.tar'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    torch.save(state, check_path)
    print('saving to {}'.format(check_path))


args = vars(get_arguments())

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = args['pretrained_weights']
    if os.path.exists(checkpoints_weight):
        pretrained = False

    dataset = VOCDataset(args['image_dir'], args['labels_dir'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args['num_workers'])
    model = CTPN_Model()
    model.to(device)
    
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']

    params_to_uodate = model.parameters()
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)
    
    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(resume_epoch+1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#'*50)
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)
    
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()
    
            out_cls, out_regr = model(imgs)
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)
    
            loss = loss_cls + loss_regr  # total loss
            loss.backward()
            optimizer.step()
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i+1
    
            print(f'Ep:{epoch}/{epochs-1}--'
                  f'Batch:{batch_i}/{epoch_size}\n'
                  f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                  f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
                  f'loss:{epoch_loss/mmp:.4f}\n')
    
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

