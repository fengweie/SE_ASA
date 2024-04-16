from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
transform_aug = transforms.Compose([transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5),
    transforms.RandomGrayscale(p=0.4)])
class CTDataset(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform=None,PL_Tag=False):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        if PL_Tag:
            self.ct_gt_list = None
        else:
            with open(gt_pth, 'r') as fp:
                self.ct_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag   # generate pseudo label or not


    def __getitem__(self, index):

        img_pth = self.ct_image_list[index][:-1]
        gt_pth  = self.ct_gt_list[index][:-1]
        img,gt  = self.load_data(img_pth,gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)

        return img, gt,index

    def __len__(self):
        return len(self.ct_image_list)
    def load_data(self,img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)  # h*w

        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt


class MRDataset(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform=None,PL_Tag=False):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()
        if PL_Tag:
            self.mr_gt_list = None

        else:
            with open(gt_pth, 'r') as fp:
                self.mr_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag

    def __getitem__(self, index):

        img_pth = self.mr_image_list[index][:-1]
        # print(len(self.mr_image_list))
        # print(len(self.mr_gt_list))
        gt_pth = self.mr_gt_list[index][:-1]
        img, gt = self.load_data(img_pth, gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)

        return img, gt, index

    def __len__(self):
        return len(self.mr_image_list)

    def load_data(self, img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3]) # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt

class CTDataset_aug(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform=None,PL_Tag=False,aug_transform=True):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        if PL_Tag:
            self.ct_gt_list = None
        else:
            with open(gt_pth, 'r') as fp:
                self.ct_gt_list = fp.readlines()
        self.aug_transform = aug_transform
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag   # generate pseudo label or not

    def __getitem__(self, index):
        img_pth = self.ct_image_list[index][:-1]
        gt_pth  = self.ct_gt_list[index][:-1]
        img,gt  = self.load_data(img_pth,gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w
        if self.aug_transform:
            img_aug_list = []
            img_aug_ori = np.load(img_pth)  # h*w*1
            img_aug_ori = np.expand_dims(img_aug_ori, -1)
            img_aug_ori = np.tile(img_aug_ori, [1, 1, 3])  # h*w*3
            img_aug_ori = (img_aug_ori + 1) * 127.5
            # img_aug = img_aug_ori.copy()
            # # img_aug = np.array(img_aug, dtype=np.uint8)
            # img_aug = img_aug[:, :, ::-1].copy()  # change to BGR
            # img_aug -= self.img_mean
            # img_aug = np.transpose(img_aug, (2, 0, 1))  # 3*h*w
            # # img_aug_ori = Image.fromarray(img_aug_ori.astype('uint8'))
            for i in range(5):
                img_aug = img_aug_ori.copy()
                img_aug = Image.fromarray(img_aug.astype('uint8'))
                img_aug = transform_aug(img_aug)
                img_aug = np.array(img_aug, dtype=np.uint8)
                img_aug = img_aug[:, :, ::-1].copy()  # change to BGR
                img_aug = img_aug - self.img_mean
                img_aug = np.transpose(img_aug, (2, 0, 1))  # 3*h*w
                img_aug_list.append(img_aug)

        gt = gt.astype(int)
        if self.aug_transform:
            return img, gt,img_aug_list,index
        else:
            return img, gt, index

    def __len__(self):
        return len(self.ct_image_list)
    def load_data(self,img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)  # h*w

        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt


class MRDataset_aug(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform=None,PL_Tag=False,aug_transform=True):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()
        if PL_Tag:
            self.mr_gt_list = None

        else:
            with open(gt_pth, 'r') as fp:
                self.mr_gt_list = fp.readlines()
        self.transform      = transform
        self.aug_transform = aug_transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag

    def __getitem__(self, index):
        img_pth = self.mr_image_list[index][:-1]
        # print(len(self.mr_image_list))
        # print(len(self.mr_gt_list))
        gt_pth = self.mr_gt_list[index][:-1]
        img, gt = self.load_data(img_pth, gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w
        if self.aug_transform:
            img_aug_list = []
            img_aug_ori = np.load(img_pth)  # h*w*1
            img_aug_ori = np.expand_dims(img_aug_ori, -1)
            img_aug_ori = np.tile(img_aug_ori, [1, 1, 3])  # h*w*3
            img_aug_ori = (img_aug_ori + 1) * 127.5
            # img_aug = img_aug_ori.copy()
            # # img_aug = np.array(img_aug, dtype=np.uint8)
            # img_aug = img_aug[:, :, ::-1].copy()  # change to BGR
            # img_aug -= self.img_mean
            # img_aug = np.transpose(img_aug, (2, 0, 1))  # 3*h*w
            # # img_aug_ori = Image.fromarray(img_aug_ori.astype('uint8'))
            for i in range(5):
                img_aug = img_aug_ori.copy()
                img_aug = Image.fromarray(img_aug.astype('uint8'))
                img_aug = transform_aug(img_aug)
                img_aug = np.array(img_aug, dtype=np.uint8)
                img_aug = img_aug[:, :, ::-1].copy()  # change to BGR
                img_aug = img_aug - self.img_mean
                img_aug = np.transpose(img_aug, (2, 0, 1))  # 3*h*w
                img_aug_list.append(img_aug)

        gt = gt.astype(int)
        if self.aug_transform:
            return img, gt, img_aug_list, index
        else:
            return img, gt, index

    def __len__(self):
        return len(self.mr_image_list)

    def load_data(self, img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3]) # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt
