
from torch import nn
from medpy.metric.binary import assd,dc
from datetime import datetime
import scipy.io as scio
import os.path as osp
import torch.backends.cudnn as cudnn
import os
import cv2
from PIL import Image
from torch.nn import functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as nd
class ECELoss(nn.Module):

    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    Acknowledge To: https://github.com/gpleiss/temperature_scaling
    """
    def __init__(self, n_bins=15, LOGIT = True):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.LOGIT = LOGIT

    def forward(self, logits, labels):
        n, c, h, w = logits.size()
        logits = logits.contiguous().transpose(1, 2).transpose(2, 3).contiguous()  # n*c*h*w->n*h*c*w->n*h*w*c
        logits = logits.view(-1, c)
        labels = labels.contiguous().view(-1)
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean().float()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
BATCHSIZE     = 32
data_size     = [256, 256, 1]
label_size    = [256, 256, 1]
NUMCLASS      = 5
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


def decode_seg_map_sequence(label_mask, dataset='pascal'):
    rgb_mask = decode_segmap(label_mask, dataset)
    rgb_masks = np.array(rgb_mask)
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
def _compute_metric(pred,target):

    pred = pred.astype(int)
    target = target.astype(int)
    dice_list  = []
    assd_list  = []
    pred_each_class_number = []
    true_each_class_number = []


    for c in range(1,NUMCLASS):
        y_true    = target.copy()
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0
        test_pred[test_pred == c] = 1
        y_true[y_true != c] = 0
        y_true[y_true == c] = 1
        pred_each_class_number.append(np.sum(test_pred))
        true_each_class_number.append(np.sum(y_true))

    for c in range(1, NUMCLASS):
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0

        test_gt = target.copy()
        test_gt[test_gt != c] = 0

        dice = dc(test_pred, test_gt)

        try:
            assd_metric = assd(test_pred, test_gt)
        except:
            print('assd error')
            assd_metric = 1

        dice_list.append(dice)
        assd_list.append(assd_metric)

    return  np.array(dice_list),np.array(assd_list)

def eval(model,testfile_list,TARGET_MODALITY,pretrained_model_pth,Method, save_img):



    dice_mean,dice_std,assd_mean,assd_std, ece_value = eval_uda(testfile_list,model, pretrained_model_pth,TARGET_MODALITY,Method, save_img)

    return dice_mean,dice_std,assd_mean,assd_std,ece_value
def _compute_entropy_map(pred):

    '''
    pred: n*c*h*w
    '''
    n,c,h,w = pred.shape
    # print(pred.shape)
    pred = torch.softmax(pred,dim=1)
    self_information_map =  -torch.mul(pred, torch.log2(pred + 1e-30)) / np.log2(c)
    entropy_map = torch.sum(self_information_map,dim=1) # n*h*w

    return entropy_map.squeeze()
def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    # print(shape)
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    min = np.amin(ent)
    max = np.amin(ent)
    return (ent - min) / 0.4
    # return (ent - min) / (max-min)
def eval_uda(testfile_list,model,pretrained_model_pth,TARGET_MODALITY,Method, save_img=False):

    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    save_images = True

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    if not osp.exists(pretrained_model_pth):
        print('')
    print('Evaluating model {}'.format(pretrained_model_pth))
    load_checkpoint_for_evaluation(model,pretrained_model_pth)


    dice_list = []
    assd_list = []

    label_all = []
    pred_all = []
    ece_metric = ECELoss()
    for idx_file, fid in enumerate(testfile_list):

        if TARGET_MODALITY == "CT":
            save_img_path = "./save_results/MR2CT/" + Method + "/img/"+str(idx_file)
            save_gt_path = "./save_results/MR2CT/" + Method + "/gt/"+str(idx_file)
            save_pred_path = "./save_results/MR2CT/" + Method + "/pred/"+str(idx_file)
            save_ent_path = "./save_results/MR2CT/" + Method + "/ent/"+str(idx_file)
        elif TARGET_MODALITY == "MR":
            save_img_path = "./save_results/CT2MR/" + Method + "/img/"+str(idx_file)
            save_gt_path = "./save_results/CT2MR/" + Method + "/gt/"+str(idx_file)
            save_pred_path = "./save_results/CT2MR/" + Method + "/pred/"+str(idx_file)
            save_ent_path = "./save_results/CT2MR/" + Method + "/ent/"+str(idx_file)
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        if not os.path.exists(save_gt_path):
            os.makedirs(save_gt_path)
        if not os.path.exists(save_pred_path):
            os.makedirs(save_pred_path)
        if not os.path.exists(save_ent_path):
            os.makedirs(save_ent_path)


        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        slice_num_img = 0
        slice_num_pred = 0
        slice_num_gt = 0
        slice_num_ent = 0
        tmp_pred = np.zeros(label.shape)

        tmp_pred_soft = np.zeros([5, 256, 256,label.shape[-1]])

        frame_list = [kk for kk in range(data.shape[2])]
        pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                item_label = label[..., jj]

                gt_save = decode_seg_map_sequence(item_label) * 255
                # print(gt.max(),gt.min(),gt.shape)
                gt_save = Image.fromarray(np.uint8(gt_save))
                if save_img:
                    gt_save.save(save_gt_path+"/slice_{}.png".format(slice_num_gt))


                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics

                img_save = Image.fromarray(((item_data + 1) * 127.5).astype('uint8'))
                if save_img:
                    img_save.save(save_img_path+"/slice_{}.png".format(slice_num_img))

                item_data = np.expand_dims(item_data, -1)

                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                data_batch[idx, ...] = item_data

                slice_num_img += 1
                slice_num_gt +=1

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                cla_feas_src,pred_b_aux, pred_b_main_soft = model(imgs)

                pred_b_main_soft = interp(pred_b_main_soft)
                pred_b_main = torch.argmax(pred_b_main_soft, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            # print(pred_b_main_soft.shape)
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()
                tmp_pred_soft[..., jj] = pred_b_main_soft[idx, ...].cpu().data.numpy().copy()
                pred_trg = decode_seg_map_sequence(pred_b_main[idx, ...].copy()) * 255
                # print(gt.max(),gt.min(),gt.shape)
                pred_trg = Image.fromarray(np.uint8(pred_trg))
                if save_img:
                    pred_trg.save(save_pred_path+"/slice_{}.png".format(slice_num_pred))
                slice_num_pred+=1

                entropy_map = _compute_entropy_map(pred_b_main_soft[idx, ...].unsqueeze(0))
                entropy_map = entropy_map.cpu().data.numpy()

                entropy_map = normalize_ent(entropy_map)
                entropy_map = construct_color_img(entropy_map)
                if save_img:
                    cv2.imwrite(save_ent_path+"/slice_{}.png".format(slice_num_ent), entropy_map)
                slice_num_ent +=1
        pred_end_time = datetime.now()
        pred_spend_time = (pred_end_time-pred_start_time).seconds
        print('pred spend time is {} seconds'.format(pred_spend_time))

        label = label.astype(int)
        metric_start_time      = datetime.now()
        dice, assd             = _compute_metric(tmp_pred,label)
        metric_end_time        = datetime.now()
        metric_spend_time      = (metric_end_time-metric_start_time).seconds
        print('metric spend time is {} seconds'.format(metric_spend_time))

        dice_list.append(dice)
        assd_list.append(assd)

        label_all.append(np.transpose(label, (2, 0, 1)))
        pred_all.append(np.transpose(tmp_pred_soft, (3,0, 1, 2)))
        print(label.shape, tmp_pred_soft.shape)
    label_all_arr = np.vstack(label_all) #N_CT * N_Class
    pred_all_arr = np.vstack(pred_all) #N_CT * N_Class
    print(label_all_arr.shape,pred_all_arr.shape)
    ece_value = ece_metric(torch.from_numpy(pred_all_arr),torch.from_numpy(label_all_arr))
    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.1f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.1f' % np.mean(assd_mean))
    print("Ece value:", ece_value)

    return dice_mean,dice_std,assd_mean,assd_std,ece_value

def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
def eval_during_train(testfile_list,model,TARGET_MODALITY):

    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()

    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
                item_data = np.expand_dims(item_data, -1)
                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                data_batch[idx, ...] = item_data

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                cla_feas_src,pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()

        pred_end_time = datetime.now()
        pred_spend_time = (pred_end_time-pred_start_time).seconds
        # print('pred spend time is {} seconds'.format(pred_spend_time))

        label = label.astype(int)
        metric_start_time      = datetime.now()
        dice, assd             = _compute_metric(tmp_pred,label)
        metric_end_time        = datetime.now()
        metric_spend_time      = (metric_end_time-metric_start_time).seconds
        # print('metric spend time is {} seconds'.format(metric_spend_time))

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)
    model.train()
    return dice_mean,dice_std,assd_mean,assd_std