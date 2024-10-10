import torchvision.transforms.functional as F
import warnings
import math
import random
import numpy as np
from PIL import Image
import torch


_pil_interpolation_to_str = {
    F.InterpolationMode.NEAREST: 'F.InterpolationMode.NEAREST',
    F.InterpolationMode.BILINEAR: 'F.InterpolationMode.BILINEAR',
    F.InterpolationMode.BICUBIC: 'F.InterpolationMode.BICUBIC',
    F.InterpolationMode.LANCZOS: 'F.InterpolationMode.LANCZOS',
    F.InterpolationMode.HAMMING: 'F.InterpolationMode.HAMMING',
    F.InterpolationMode.BOX: 'F.InterpolationMode.BOX',
}

def _pil_interp(method):
    if method == 'bicubic':
        return F.InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return F.InterpolationMode.LANCZOS
    elif method == 'hamming':
        return F.InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return F.InterpolationMode.BILINEAR

class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, augmentation=False, box=None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if augmentation:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            img = F.crop(img, i, j, h, w)
            # img, box = crop(img, i, j, h, w, box)
        img = F.resize(img, self.size, self.interpolation)
        second_img = F.resize(img, self.second_size, self.second_interpolation) \
            if self.second_size is not None else None
        return img, second_img

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
        format_string += ')'
        return format_string
