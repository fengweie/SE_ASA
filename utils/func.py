import numpy as np
import torch
import torch.nn as nn
from utils.loss import cross_entropy_2d
# import cv2
import torch.nn.functional as F
import torch.sparse as sparse
from skimage.exposure import match_histograms
def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    each_class_number = []
    eps      = 1e-7
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number)

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def loss_calc(pred,label,cfg):

    '''
    This function returns cross entropy loss for semantic segmentation
    '''
    # pred shape is batch * c * h * w
    # label shape is b*h*w
    label = label.long().cuda()
    return cross_entropy_2d(pred, label,cfg)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)

def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)

def sel_prob_2_entropy(prob):
    n, c, h, w = prob.size()
    weighted_self_info = -torch.mul(prob, torch.log2(prob + 1e-30)) / c
    entropy            = torch.sum(weighted_self_info,dim=1) #N*C*H*W
    # mean_entropy       = torch.sum(entropy,dim=[1,2])
    return entropy



def mpcl_loss_calc(feas,labels,class_center_feas,loss_func,
                               pixel_sel_loc=None,tag='source'):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    n,c,fea_h,fea_w = feas.size()
    if tag == 'source':
        labels      = labels.float()
        labels      = F.interpolate(labels, size=fea_w, mode='nearest')
        labels      = labels.permute(0,2,1).contiguous()
        labels      = F.interpolate(labels, size=fea_h, mode='nearest')
        labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w

    labels  = labels.cuda()
    labels  = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas,p=2,dim=1)
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss =  loss_func(feas,labels,class_center_feas,
                                                    pixel_sel_loc=pixel_sel_loc)
    return loss

from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt
from torchvision import transforms

# from  A Fourier-based Framework for Domain Generalization
def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    # img21 = np.uint8(np.clip(img21, 0, 255))
    # img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


# from FedDG
def amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    # deep copy
    a_local_copy = a_local.copy()
    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_trg[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2] * ratio + a_local_copy[:, h1:h2, w1:w2] * (1 - ratio)

    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    a_trg = np.fft.ifftshift(a_trg, axes=(-2, -1))
    return a_local, a_trg


def freq_space_interpolation(local_img, trg_img, L=0, ratio=0):
    local_img_np = local_img
    tar_img_np = trg_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(tar_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_, amp_trg_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    fft_trg_ = amp_trg_ * np.exp(1j * pha_trg)
    trg_in_local = np.fft.ifft2(fft_trg_, axes=(-2, -1))
    trg_in_local = np.real(trg_in_local)

    return local_in_trg, trg_in_local


# i is the lambda of target
def fourier_transform(im_local, im_trg, L=0.01, i=1):
    im_local = im_local.transpose((2, 0, 1)) ### 1x256x256
    im_trg = im_trg.transpose((2, 0, 1))### 1x256x256
    local_in_trg, trg_in_local = freq_space_interpolation(im_local, im_trg, L=L, ratio=1 - i)
    local_in_trg = local_in_trg.transpose((1, 2, 0))  ####256x256x1
    trg_in_local = trg_in_local.transpose((1, 2, 0))   ####256x256x1
    return local_in_trg, trg_in_local


def save_image(image, path):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)
    # plt.show()
    return 0
def fourier_augmentation(img, tar_img, mode, alpha):
    # transfer image from PIL to numpy
    img = np.array(img)
    tar_img = np.array(tar_img)
    img = img[:,:,np.newaxis]
    tar_img = tar_img[:,:,np.newaxis]

    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    if mode == 'AS':
        # print("using AS mode")
        aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.01, i=1)
    elif mode == 'AM':
        # print("using AM mode")
        aug_img, aug_tar_img = colorful_spectrum_mix(img, tar_img, alpha=alpha)
    else:
        print("mode name error")

    aug_img = np.squeeze(aug_img)  ####256x256
    aug_img = Image.fromarray(aug_img)  ####256x256

    aug_tar_img = np.squeeze(aug_tar_img)    ####256x256
    aug_tar_img = Image.fromarray(aug_tar_img)  ####256x256

    return aug_img, aug_tar_img

if __name__ == '__main__':
    mri_img2 = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/train_ct/ct_train_slice45.npy'
    ct_img1 = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/train_mr/mr_train_slice4185.npy'
    ct_mask1 =  '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/gt_train_ct/ct_train_slice45_gt.npy'
    mri_mask2 =  '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/gt_train_mr/mr_train_slice4185_gt.npy'

    image_size=256
    img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.7, contrast=0.9, saturation=0.9, hue=0.5),
    transforms.RandomGrayscale(p=0.5)
            ])
    img_mean =   np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    # mask1 = Image.open(ct_mask1)
    # mask2 = Image.open(mri_mask2)
    # mask1 = img_transform(mask1)
    # mask2 = img_transform(mask2)
    # mask1 = np.asarray(mask1)
    # mask2 = np.asarray(mask2)

    ct_img1 = np.load(ct_img1)
    mri_img2 = np.load(mri_img2)

    ct_img1 = np.asarray(ct_img1)
    # ct_img1 = ct_img1[:,:,np.newaxis]
    ct_img1 = (ct_img1 + 1) * 127.5
    # # ct_img1 = (ct_img1 + 1) * 127.5
    # # mri_img2 = (mri_img2 + 1) * 127.5
    # # ct_img_match_his = match_histograms(ct_img1, mri_img2)
    # # save_image((ct_img_match_his / 255), '/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_im_local_match_his.jpg')
    # print(ct_img1.max(),ct_img1.min())
    # # print(mri_img2.max(),mri_img2.min())
    # save_image((ct_img1 / 255), '/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_im_local.png')
    #
    # print(ct_img1.max(),ct_img1.min(),ct_img1.astype(np.uint8).max(),ct_img1.astype(np.uint8).min())
    # ct_img1 = Image.fromarray(ct_img1.astype(np.uint8))
    # mri_img2 = Image.fromarray(mri_img2.astype(np.uint8))
    # ct_img1 = img_transform(ct_img1)
    #
    # ct_img1 = np.asarray(ct_img1)
    # mri_img2 = np.asarray(mri_img2)
    #
    # save_image((ct_img1 / 255),'/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_aug_local.png')





    # save_image((aug_local / 255),
    #            '/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_aug_local.jpg')
    # mri_img2 = img_transform(mri_img2)


    # ct_img1 = ct_img1[:, :, ::-1].copy()  # change to BGR
    # ct_img1 -= img_mean
    # mri_img2 = np.tile(mri_img2, [1, 1, 3])  # h*w*3
    mri_img2 = (mri_img2 + 1) * 127.5
    # mri_img2 = mri_img2[:, :, np.newaxis]
    # mri_img2 = mri_img2[:, :, ::-1].copy()  # change to BGR
    # mri_img2 -= img_mean
    print(ct_img1.shape)
    print(mri_img2.shape)

    im_local = ct_img1
    im_trg = mri_img2

    # image = cv2.cvtColor((image.numpy().transpose(
    #     1, 2, 0) + 1) * 127.5, cv2.COLOR_BGR2RGB).astype(np.uint8)
    transform_aug = transforms.Compose([transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
                                        transforms.RandomGrayscale(p=0.7)])
    aug_local = transform_aug(Image.fromarray(im_local.astype(np.uint8)))
    aug_local = np.asarray(aug_local)
    aug_traget = im_trg
    # aug_local, aug_traget = fourier_transform(im_local, im_trg, L=0.0125, i=1)
    print(aug_local.max(),aug_local.min(), aug_traget.max(),aug_traget.min())
    # aug_local, aug_traget = colorful_spectrum_mix(im_local, im_trg, alpha=0.5)
    save_image((im_local / 255),'/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_im_local.png')
    save_image((im_trg / 255),'/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/mri_im_trg.png')
    save_image((aug_local / 255),'/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/ct_aug_local.png')
    save_image((aug_traget / 255),'/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/utils/fourier_img/mri_aug_traget.png')
