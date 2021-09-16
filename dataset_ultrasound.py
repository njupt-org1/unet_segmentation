import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from elastic_transform import elastic_transform, affine_elastic_transform
import cv2


class UltraSoundSingleDataset(Dataset):
    def __init__(self, dir_imgs: Path, dir_masks: Path, scale: float = 1):
        self.dir_imgs = dir_imgs
        self.dir_masks = dir_masks
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.paths_img = sorted(dir_imgs.iterdir(), key=lambda x: x.stem)
        self.paths_mask = sorted(dir_masks.iterdir(), key=lambda x: x.stem)

        self.check_img_with_mask()
        logging.info(f'Creating dataset with {len(self.paths_img)} examples')

    def __len__(self):
        return len(self.paths_img)

    def check_img_with_mask(self):
        """image and mask should have same file name."""
        len_imgs = len(self.paths_img)
        len_masks = len(self.paths_mask)
        collect_not_match = []
        assert len_imgs == len_masks, \
            f"The number of images ({len_imgs}) does not match the number of masks ({len_masks})."
        for id_img, id_mask in zip(self.paths_img, self.paths_mask):
            if id_img.stem != id_mask.stem:
                collect_not_match.append((id_img.stem, id_mask.stem))

        assert collect_not_match == [], f'Exist mismatch pairs: {collect_not_match}'

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small'
        pil_img = pil_img.resize((new_w, new_h))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        path_img = self.paths_img[i]
        path_mask = self.paths_mask[i]

        img = Image.open(path_img).convert('L')
        mask = Image.open(path_mask)

        assert img.size == mask.size, \
            f'Image and mask ({path_img.stem}) should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class UltraSoundMultiDataset(Dataset):
    def __init__(self, dirs_img: List[Path], dirs_mask: List[Path],
                 size_expect: tuple = (500, 400), scale: float = 1, transforms: List[str] = []):
        self.dirs_img = dirs_img
        self.dirs_mask = dirs_mask
        self.scale = scale
        self.wh_expect = size_expect
        self.transforms = transforms
        self.transforms_available = ['flip', 'rotate', 'elastic', 'affine_elastic', 'gaussian']
        self._check_transforms()
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.paths_img, self.paths_mask = self.load_paths_per_dataset()

    def __len__(self):
        return len(self.paths_img)

    def _check_transforms(self):
        if self.transforms not in [None, []]:
            for t in self.transforms:
                assert t in self.transforms_available, \
                    f'Operation {t} is not in {self.transforms_available}!'

    def load_paths_per_dataset(self):
        paths_img_all = []
        paths_mask_all = []
        for i, (dir_img, dir_mask) in enumerate(zip(self.dirs_img, self.dirs_mask), 1):
            paths_img = sorted(dir_img.iterdir(), key=lambda x: x.stem)
            paths_mask = sorted(dir_mask.iterdir(), key=lambda x: x.stem)
            self.check_img_with_mask(paths_img, paths_mask)
            logging.info(f'Dataset-{i: <2}:{len(paths_img)} examples')
            paths_img_all += paths_img
            paths_mask_all += paths_mask
        logging.info(f'All Datasets: {len(paths_img_all)} examples')
        return paths_img_all, paths_mask_all

    @staticmethod
    def check_img_with_mask(paths_img, paths_mask):
        """image and mask should have same file name."""
        len_imgs = len(paths_img)
        len_masks = len(paths_mask)
        collect_not_match = []
        assert len_imgs == len_masks, \
            f"The number of images ({len_imgs}) does not match the number of masks ({len_masks})."
        for id_img, id_mask in zip(paths_img, paths_mask):
            if id_img.stem != id_mask.stem:
                collect_not_match.append((id_img.stem, id_mask.stem))

        assert collect_not_match == [], f'Exist mismatch pairs: {collect_not_match}'

    @staticmethod
    def resize_with_padding(in_img: Image, size_out: tuple, pad_style: str = 'average'):
        def get_img_new_wh(old_wh: tuple):
            old_w, old_h = old_wh
            dst_w, dst_h = size_out  # include pad area

            if old_w / old_h > dst_w / dst_h:
                new_h = dst_h
                new_w = new_h * old_w / old_h
            else:
                new_w = dst_w
                new_h = new_w * old_h / old_w

            return round(new_w), round(new_h)

        # ------ choose pad color ------
        if pad_style == 'average':
            pad_color = in_img.resize((1, 1)).getpixel((0, 0))
        elif pad_style == 'black':
            if in_img.mode in ['L', 'P']:
                pad_color = 0
            elif in_img.mode == 'RGB':
                pad_color = (0, 0, 0)
            elif in_img.mode == '1':
                pad_color = False
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        out_img = Image.new(in_img.mode, size_out, pad_color)
        in_img_scaled = in_img.resize(get_img_new_wh(in_img.size))
        upper_left_corner = ((out_img.size[0] - in_img_scaled.size[0]) // 2,
                             (out_img.size[1] - in_img_scaled.size[1]) // 2)
        out_img.paste(in_img_scaled, upper_left_corner)

        return out_img

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small'
        pil_img = pil_img.resize((new_w, new_h))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def show_samples(self, num=5):
        idx_random = np.random.choice(self.__len__(), num, replace=False)
        for n, idx in enumerate(idx_random):
            pair = self.__getitem__(idx)
            image = pair['image'].numpy()[0] * 255
            mask = pair['mask'].numpy()[0] * 255
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            dst = Image.new('L', (image.width + mask.width, image.height))
            dst.paste(image, (0, 0))
            dst.paste(mask, (image.width, 0))

            dst.show(str(n))

    def show_img_with_mask_outline(self, idx):
        pair = self.__getitem__(idx)
        image = pair['image'].numpy()[0] * 255
        image = image.astype(np.uint8)
        cv2.imshow('image', image)
        mask = pair['mask'].numpy().astype(np.uint8).squeeze()
        # https://docs.opencv.org/4.5.3/d4/d73/tutorial_py_contours_begin.html
        # findContours need 0-1 image input
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # -1 means draw all the contours
        cv2.drawContours(image, contours, -1, color=(255, 0, 0), thickness=1)

        cv2.imshow('image_contour', image)
        cv2.waitKey()

    def _do_transforms(self, image, mask):
        if random.random() < 0.5:
            return image, mask

        if 'flip' in self.transforms:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if 'rotate' in self.transforms:
            degree = random.randint(0, 10)
            image = TF.rotate(image, angle=degree)
            mask = TF.rotate(mask, angle=degree)

        if 'elastic' in self.transforms:
            alpha = random.uniform(300, 400)  # 200
            sigma = random.uniform(9, 13)  # 9-13
            image, mask = elastic_transform(image, mask, alpha, sigma)

        if 'affine_elastic' in self.transforms:
            alpha = random.uniform(0, 200)
            sigma = random.uniform(9, 13)
            alpha_affine = random.uniform(0, 30)
            image, mask = affine_elastic_transform(image, mask,
                                                   alpha, sigma, alpha_affine)

        if 'gaussian' in self.transforms:
            image = np.array(image)
            assert len(image.shape) == 2
            mean = 8
            var = 10
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, image.shape)  # np.zeros((224, 224), np.float32)
            noisy_image = image + gaussian
            noisy_image = np.clip(noisy_image, 0, 255)
            image = Image.fromarray(noisy_image.astype('uint8'))

        return image, mask

    def __getitem__(self, i):
        path_img = self.paths_img[i]
        path_mask = self.paths_mask[i]

        img = Image.open(path_img).convert('L')
        mask = Image.open(path_mask).convert('P')
        assert img.size == mask.size, \
            f'Image and mask ({path_img.stem}) should be the same size, but are {img.size} and {mask.size}'

        if self.transforms is not None:
            img, mask = self._do_transforms(img, mask)

        if img.size != self.wh_expect:
            img = self.resize_with_padding(img, self.wh_expect, pad_style='average')
            mask = self.resize_with_padding(mask, self.wh_expect, pad_style='black')

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


if __name__ == '__main__':
    dir_train_img_1 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\gray')
    dir_train_mask_1 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\\{t}\mask')
    dir_train_img_2 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\malignant\gray')
    dir_train_mask_2 = Path('C:\\Users\Administrator\Desktop\智能医疗\数据集\BUSI_split_merge\malignant\mask')
    dirs_img = [dir_train_img_1]
    dirs_mask = [dir_train_mask_1]
    ds = UltraSoundMultiDataset(dirs_img, dirs_mask, transforms=['elastic'])
    o = ds[3]
    # ds.show_samples(1)
    for i in range(10, 20):
        ds.show_img_with_mask_outline(i)
