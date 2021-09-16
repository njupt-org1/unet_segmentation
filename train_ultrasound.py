import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from dice_loss import dice_coeff
from eval import eval_net
from unet import UNet

from dataset_ultrasound import UltraSoundSingleDataset, UltraSoundMultiDataset
from torch.utils.data import DataLoader, random_split


def get_ds(img_scale, aug_list):
    # ====== original training data ======
    #dir_train_img_1 = Path('/home/gy/ultrasound_dataset/T_BUSIS/train_gray')
    #dir_train_mask_1 = Path('/home/gy/ultrasound_dataset/T_BUSIS/train_mask')
    dir_train_img_1 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\乳腺肿瘤分割数据（赵）\原始数据集train_(L)')
    dir_train_mask_1 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\乳腺肿瘤分割数据（赵）\原始数据集_train_masks_format')

    # ====== other dataset ======
    t = 'benign'  # 'benign', 'malignant' or 'normal'
    #dir_train_img_2 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/gray')
    #dir_train_mask_2 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/mask')
    dir_train_img_2 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\gray')
    dir_train_mask_2 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\mask')
    t = 'malignant'
    #dir_train_img_3 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/gray')
    #dir_train_mask_3 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/mask')
    dir_train_img_3 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\gray')
    dir_train_mask_3 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\mask')
    t = 'normal'
    #dir_train_img_4 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/gray')
    #dir_train_mask_4 = Path(f'/home/gy/ultrasound_dataset/BUSI/gray_mask_split/{t}/mask')
    dir_train_img_4 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\gray')
    dir_train_mask_4 = Path(f'C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\mask')

    # ====== select dataset to train ======
    dirs_img_train = [dir_train_img_1, ]
    dirs_mask_train = [dir_train_mask_1, ]
    ds_train = UltraSoundMultiDataset(dirs_img_train, dirs_mask_train, (500, 400), img_scale,
                                      transforms=aug_list)  # ['flip', 'rotate', 'elastic',]

    # ====== only use original test data to evaluate ======
    #dir_val_img = Path('/home/gy/ultrasound_dataset/T_BUSIS/test_gray')
    #dir_val_mask = Path('/home/gy/ultrasound_dataset/T_BUSIS/test_mask')
    dir_val_img = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\乳腺肿瘤分割数据（赵）\\breast_cancer_test(L)')
    dir_val_mask = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\乳腺肿瘤分割数据（赵）\\breast_cancer_test_masks_format')
    ds_val = UltraSoundSingleDataset(dir_val_img, dir_val_mask, img_scale)

    return ds_train, ds_val


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              dir_ckpt=None,
              img_scale=0.5,
              aug_list=[],
              str_result=''):
    ds_train, ds_val = get_ds(img_scale, aug_list)

    path_val_score = Path(f'val_score/{str_result}')
    path_val_score.parent.mkdir(exist_ok=True)
    assert not path_val_score.with_suffix('.npy').exists()

    n_train = len(ds_train)
    n_val = len(ds_val)
    # Todo: random_split?
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=False)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {dir_ckpt}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.1)#学习率调整策略
    # scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss() +
    save_val_score = []

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                # loss_dice = 1 - dice_coeff(masks_pred, true_masks)
                loss_ce = criterion(masks_pred, true_masks)
                loss = loss_ce
                #return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)#梯度截断,梯度值大于0.1就设为0.1，小于-0.1就设为-0.1
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
        # if global_step % (n_train // (10 * batch_size)) == 0:
        val_score = eval_net(net, val_loader, device)
        # todo: do not change lr?
        # scheduler1.step(val_score)
        scheduler2.step()
        # scheduler3.step(val_score)
        save_val_score.append(val_score)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))

        if dir_ckpt is not None:
            dir_ckpt = Path(dir_ckpt)
            dir_ckpt.mkdir(exist_ok=True, parents=True)

            torch.save(net.state_dict(),
                       dir_ckpt.joinpath(f'epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    save_val_score = np.array(save_val_score)
    np.save(path_val_score, save_val_score)


@dataclass
class Param:
    epochs: int = 5
    batch_size: int = 1
    lr: float = 0.001
    scale: float = 0.5
    dir_ckpt: str = 'checkpoints/'


def set_seed(seed: int = 0):
    random.seed(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(aug_list, result_name):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = Param(
        epochs=100,
        batch_size=5,
        lr=0.0001,
        scale=1,
        # dir_ckpt='checkpoints/',
        dir_ckpt=None,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    set_seed(0)
    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  dir_ckpt=args.dir_ckpt,
                  aug_list=aug_list,
                  str_result=result_name)
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        # logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    # aug_list = ['flip', 'rotate', 'elastic', 'gaussian']
    # for a in aug_list:
    #     main([a], a)
    main(['flip'], 'instancenorm_leak')
