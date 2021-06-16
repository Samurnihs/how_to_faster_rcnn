import random
import torch
from torchvision.transforms import functional as F
import math
import numbers
import warnings
from PIL import Image


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = image.size[0]
            image = image.transpose(Image.FLIP_LEFT_RIGHT) 
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height = image.size[1]
            image = image.transpose(Image.FLIP_TOP_BOTTOM) 
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, resample=False, expand=False, center=None, fill=None):
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill


    @staticmethod
    def get_params():
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice([0, 90, 180, 270])

        return angle
    
    @staticmethod
    def rotate_tensor_box(boxes, angle, size):
    
        rads = angle / 180 * math.pi
        sin = math.sin(rads)
        cos = math.cos(rads)
    
        rotation_matrix = torch.as_tensor([[cos,  -sin], [sin, cos]], dtype=torch.float32)
    
        bias = torch.as_tensor([[size[0], size[1]]]* boxes.shape[0], dtype=torch.float32) / 2
    
        coord1 = torch.as_tensor(boxes[:, :2], dtype=torch.float32)
        coord2 = torch.as_tensor(boxes[:, 2:], dtype=torch.float32)
    
        centers = (coord1 + coord2)/2
        diff = (coord2 - coord1)/2
        
        centers_rot = torch.mm((centers - bias), rotation_matrix) + bias
        
        rectangles = torch.cat(((centers_rot - diff), (centers_rot + diff)), 1)
        
        return torch.as_tensor(rectangles, dtype=torch.int32)
        

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params()
        target['boxes'] = self.rotate_tensor_box(target['boxes'], angle, img.size)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill), target

    def __repr__(self):
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
