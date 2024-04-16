
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torchvision import transforms
from PIL import Image
# transform_aug = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#     ])
# img_mean=np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
# train_mr_data_pth = '/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/data/datalist/train_mr.txt'
# train_mr_gt_pth   = '/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/data/datalist/train_mr_gt.txt'
# def load_data(img_pth, gt_pth):
#     img = np.load(img_pth)  # h*w*1
#     gt = np.load(gt_pth)  # h*w
#
#     img = np.expand_dims(img, -1)
#     img = np.tile(img, [1, 1, 3])  # h*w*3
#     img = (img + 1) * 127.5
#     img = img[:, :, ::-1].copy()  # change to BGR
#     img -= img_mean
#     return img, gt
#
#
# with open(train_mr_data_pth, 'r') as fp:
#     mr_image_list = fp.readlines()
#
# with open(train_mr_gt_pth, 'r') as fp:
#     mr_gt_list = fp.readlines()
# img_pth = mr_image_list[100][:-1]
# # print(len(self.mr_image_list))
# # print(len(self.mr_gt_list))
# gt_pth = mr_gt_list[100][:-1]
# img, gt = load_data(img_pth, gt_pth)
#
#
# img = np.transpose(img, (2, 0, 1))  # 3*h*w
#
# img_aug_list = []
# img_aug_ori = np.load(img_pth)  # h*w*1
# img_aug_ori = np.expand_dims(img_aug_ori, -1)
# img_aug_ori = np.tile(img_aug_ori, [1, 1, 3])  # h*w*3
# img_aug_ori = (img_aug_ori + 1) * 127.5
#
#
# for i in range(5):
#     img_aug = img_aug_ori.copy()
#     img_aug = Image.fromarray(img_aug.astype('uint8'))
#     img_aug = transform_aug(img_aug)
#     img_aug = np.array(img_aug, dtype=np.uint8)
#     img_aug = img_aug[:, :, ::-1].copy()  # change to BGR
#     img_aug = img_aug - img_mean
#     img_aug = np.transpose(img_aug, (2, 0, 1))  # 3*h*w
#     #
#     img_aug_list.append(img_aug)
#
# gt = gt.astype(int)
TRAIN_RANDOM_SEED = 1234
print("training seed used:", TRAIN_RANDOM_SEED)

torch.manual_seed(TRAIN_RANDOM_SEED)
torch.cuda.manual_seed(TRAIN_RANDOM_SEED)
torch.cuda.manual_seed_all(TRAIN_RANDOM_SEED)  # 为所有GPU设置随机种子
np.random.seed(TRAIN_RANDOM_SEED)
random.seed(TRAIN_RANDOM_SEED)
print("aaa")
# pred_trg_main = torch.randn(2,3,2)
# # print(pred_trg_main[1,:,1])
# log_q = torch.FloatTensor([1,2,3])
# # print(log_q)
# # print(pred_trg_main.softmax(dim=1)[1,:,1])
# print(pred_trg_main.softmax(dim=1))
# # print(pred_trg_main.softmax(dim=1).view(-1,pred_trg_main.shape[1]))
# x = pred_trg_main.softmax(dim=1) * log_q.reshape(1, 3, 1).repeat(pred_trg_main.shape[0], 1, pred_trg_main.shape[2])
# # print(log_q.reshape(1, 3, 1).repeat(pred_trg_main.shape[0], 1, pred_trg_main.shape[2]))
# print(x)
# # print(x.view(-1,pred_trg_main.shape[1]))
# # print(pred_trg_main.softmax(dim=1).view(-1,pred_trg_main.shape[1]) *
# #                       log_q.reshape(1,3))
##################################################################################################
# consistent_idxs = torch.tensor([True, False]).repeat(2,1)
# print(consistent_idxs)
# consistent_idxs_x = consistent_idxs.clone().reshape(-1)  ###BH
# score_t_aug_curr = torch.randn(2,3,2)
# print(score_t_aug_curr)
# score_t_aug_curr_x = score_t_aug_curr.clone().reshape(-1, score_t_aug_curr.shape[1])  ###BH*C
# print(score_t_aug_curr_x)
# print(score_t_aug_curr_x[consistent_idxs_x, :])  ###BHWxc
# print((score_t_aug_curr*consistent_idxs.unsqueeze(1).repeat(1,3,1)).shape)
# print(score_t_aug_curr[0,:,0],score_t_aug_curr[1,:,0])
# score_t_aug_pos = torch.zeros(2,3,2)
# score_t_aug_pos = torch.where(consistent_idxs.unsqueeze(1).repeat(1,3,1), score_t_aug_curr, score_t_aug_pos)
# print(score_t_aug_pos)
# print(torch.masked_select(score_t_aug_curr, consistent_idxs.unsqueeze(1).repeat(1,3,1)))
# # score_t_aug_neg[inconsistent_idxs_x, :] = score_t_aug_curr_x[inconsistent_idxs_x, :]  ####BH*C

score_t_aug_curr = torch.randn(2,3,2)
# print(score_t_aug_curr)
score_t_aug_curr = score_t_aug_curr.transpose(0,1)
print(score_t_aug_curr)
cla_feas_trg_de     = torch.reshape(score_t_aug_curr,[-1,2])
print(cla_feas_trg_de)
cla_feas_trg_de1     = torch.reshape(cla_feas_trg_de,[3,2,2])
print(cla_feas_trg_de1)
cla_feas_trg_de1     = torch.reshape(score_t_aug_curr,[2,-1])
print(cla_feas_trg_de1[:,0])
