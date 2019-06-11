import torch
import math
import numbers
import random
import numpy as np
import pdb

from PIL import Image, ImageOps, ImageEnhance


class RandomHorizontalFlip(object):
    """
    Flip the image and mask from top to bottom or from left to right
    Args:
        sample: include image and label
    Returns:
        image and label (dict)
    """

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': img, 'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return {'image': img, 'label': mask}


def to_one_hot(mask, n_class):
    """
    Transform a mask to one hot
    Args:
        mask:
        n_class: number of class for segmentation
    Returns:
        y_one_hot: one hot mask
    """
    y_one_hot = torch.zeros((n_class, mask.shape[1], mask.shape[2]))
    y_one_hot = y_one_hot.scatter(0, mask, 1).long()
    return y_one_hot


def get_class_label(mask, n_class):
    """
    Get the class from the labeled mask
    Args:
    Returns:
        y: the class label with shape (n_class,)
        
    """
    y = torch.zeros((n_class, ))
    for i in range(n_class):
        if torch.sum(mask == i + 1) > 0:
            y[i] = 1
    return y


class ToTensor2(object):
    def __init__(self, n_class):
        self.n_class = n_class

    """
    Convert ndarrays in sample to Tensors.
    ########################## worth noting ##################################
    Because the multi-label classification using 4 bce loss function while
    the segmentation using 5 one-hot coding, so for multi-label classification,
    n_class = 4, while for segmentation, n_class = 5
    ########################## worth noting ##################################
    Args:
        n_class:
        sample:
    Returns:
        a dict
        label_s: the one hot mask
        label_c: the class for multi-label classification
    
    """

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(
            np.array(sample['label']).astype(np.float32), -1).transpose(
                (2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(np.round(mask)).long()
        c_label = get_class_label(mask, self.n_class - 1)
        mask = to_one_hot(mask, self.n_class)
        return {'image': img, 'label_s': mask, 'label_c': c_label}


class FixedResize(object):
    """
    Fixed size for testing
    Args:
        size: image size with width and height
    """

    def __init__(self, size):
        self.size = tuple(reversed(size))

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size  # target size
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.BILINEAR)
        return {'image': img, 'label': mask}


class RandomRotate(object):
    """
    Random rotate the image and label
    Args:
        degree: the rotate degree from (-degree , degree)
    Returns:
        rotated image and mask
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return {'image': img, 'label': mask}


class RandomZoom(object):
    """
    The major augmentation function.
    We first crop the image and mask and then zoom in them to fixed size
    Args:
        size: the fixed size for zooming in
        crop_rate: the crop rate for crop, the inital value is (0.6, 1.). we 
        suggest his should be corresponding to multi-scale input validation.
        
    """

    def __init__(self, size):
        assert type(size) is tuple
        self.size = size

    def __call__(self, sample, crop_rate=(0.6, 1.)):
        img = sample['image']
        mask = sample['label']
        fixed_rate = random.uniform(crop_rate[0], crop_rate[1])
        h, w = img.size[0], img.size[1]  # source image width and height
        tw, th = int(self.size[0] * fixed_rate), int(
            self.size[1] * fixed_rate)  #croped width and height

        if fixed_rate < 1.:
            left_shift = []
            mask_np = np.round(np.array(mask))
            select_index = np.concatenate([np.where(mask_np != 0)], axis=1)
            if select_index.shape[1] == 0:
                left_shift.append([0, (w - tw)])
                left_shift.append([0, (h - th)])
            else:
                x_left = max(0, min(select_index[0]))
                x_right = min(w, max(select_index[0]))
                y_left = max(0, min(select_index[1]))
                y_right = min(h, max(select_index[1]))
                left_shift.append(
                    [max(0, min(x_left, x_right - tw)),
                     min(x_left, w - tw)])
                left_shift.append(
                    [max(0, min(y_left, y_right - th)),
                     min(y_left, h - th)])
            x1 = random.randint(left_shift[1][0], left_shift[1][1])
            y1 = random.randint(left_shift[0][0], left_shift[0][1])
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        else:
            pw, ph = tw - w, th - h
            pad_value = [
                int(random.uniform(0, pw / 2)),
                int(random.uniform(0, ph / 2))
            ]
            img = ImageOps.expand(img,
                                  border=(pad_value[0], pad_value[1],
                                          tw - pad_value[0],
                                          th - pad_value[1]),
                                  fill=0)
            mask = ImageOps.expand(mask,
                                   border=(pad_value[0], pad_value[1],
                                           tw - pad_value[0],
                                           th - pad_value[1]),
                                   fill=0)
        tw, th = self.size[0], self.size[1]
        img, mask = img.resize((tw, th), Image.BILINEAR), mask.resize(
            (tw, th), Image.NEAREST)
        sample = {'image': img, 'label': mask}
        return sample
    
    