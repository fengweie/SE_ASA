# import numpy as  np
# from PIL import Image
# import torch
# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
#            190, 153, 153, 153, 153, 153, 250,
#            170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152,
#            70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0,
#            142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)
#
#
# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask
#
#
# def make_palette(num_classes):
#     """
#     Maps classes to colors in the style of PASCAL VOC.
#     Close values are mapped to far colors for segmentation visualization.
#     See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
#     Takes:
#         num_classes: the number of classes
#     Gives:
#         palette: the colormap as a k x 3 array of RGB colors
#     """
#     palette = np.zeros((num_classes, 3), dtype=np.uint8)
#     for k in range(0, num_classes):
#         label = k
#         i = 0
#         while label:
#             palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
#             palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
#             palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
#             label >>= 3
#             i += 1
#     return palette
#
# palette = make_palette(5)
#
#
# def color_seg(seg, from_numpy=False):
#     """
#     Replace classes with their colors.
#     Takes:
#         seg: H x W segmentation image of class IDs
#     Gives:
#         H x W x 3 image of class colors
#     """
#     if not from_numpy:
#         seg_vis = seg.detach().cpu().numpy()
#
#     p =  palette[seg_vis.flat].reshape(seg_vis.shape + (3,))
#     if not from_numpy:
#         return torch.from_numpy(p).permute((2, 0, 1))
#     else:
#         return np.transpose(p, [2, 0, 1])
#
#
# def vis_seg(img, seg, alpha=0.5):
#     """
#     Visualize segmentation as an overlay on the image.
#     Takes:
#         img: H x W x 3 image in [0, 255]
#         seg: H x W segmentation image of class IDs
#         palette: K x 3 colormap for all classes
#         alpha: opacity of the segmentation in [0, 1]
#     Gives:
#         H x W x 3 image with overlaid segmentation
#     """
#     vis = np.array(img, dtype=np.float32)
#     mask = seg > 0
#     vis[mask] *= 1. - alpha
#     vis[mask] += alpha * palette[seg[mask].flat]
#     vis = vis.astype(np.uint8)
#     return vis

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as nd

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
