import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from skimage.exposure import match_histograms
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.func import dice_eval
from model.discriminator import get_discriminatord
from utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.func import loss_calc, bce_loss
from utils.loss import dice_loss,MPCL,softmax_kl_loss,softmax_mse_loss
from utils.func import prob_2_entropy,mpcl_loss_calc,fourier_augmentation,fourier_transform
from utils.viz_segmask import decode_seg_map_sequence
from domain_adaptation.eval_UDA import eval_during_train
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

plt.switch_backend("agg")

interp_up = nn.Upsample(size=(256, 256), mode='bilinear',
                        align_corners=True)
def compute_prf1(true_mask, pred_mask):
    """
    Compute precision, recall, and F1 metrics for predicted mask against ground truth
    """
    conf_mat = confusion_matrix(true_mask.reshape(-1), pred_mask.reshape(-1), labels=[False, True])
    p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
    r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return conf_mat, p, r, f1
def generate_pseudo_label(cla_feas_trg,class_centers,cfg):

    '''
    class_centers: C*N_fea
    cla_feas_trg: N*N_fea*H*W
    '''
    cla_feas_trg_de     = cla_feas_trg.detach()
    # class_centers_norm  = class_centers.detach()
    batch,N_fea,H,W     = cla_feas_trg_de.size()
    cla_feas_trg_de     = interp_up(cla_feas_trg_de)
    cla_feas_trg_de     = F.normalize(cla_feas_trg_de,p=2,dim=1)
    class_centers_norm  = F.normalize(class_centers,p=2,dim=1)
    cla_feas_trg_de     = cla_feas_trg_de.permute(0, 2, 3,1) # N*H*W*N_fea
    cla_feas_trg_de     = torch.reshape(cla_feas_trg_de,[-1,N_fea])
    class_centers_norm  = class_centers_norm.transpose(0,1)  # N_fea*C

    batch_pixel_cosine  = torch.matmul(cla_feas_trg_de,class_centers_norm) #N*N_class
    threshold = cfg.TRAIN.PIXEL_SEL_TH
    pixel_mask          = pixel_selection(batch_pixel_cosine,threshold)
    hard_pixel_label    = torch.argmax(batch_pixel_cosine,dim=1)

    return hard_pixel_label,pixel_mask

def pixel_selection(batch_pixel_cosine,th):
    one_tag = torch.ones([1]).float().cuda()
    zero_tag = torch.zeros([1]).float().cuda()

    batch_sort_cosine,_ = torch.sort(batch_pixel_cosine,dim=1)
    pixel_sub_cosine    = batch_sort_cosine[:,-1]-batch_sort_cosine[:,-2]
    # pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)
    pixel_mask          = pixel_sub_cosine > th
    return pixel_mask

def iter_eval(model,images_sval,labels_sval,images_target,labels_target,cfg):
    model.eval()

    with torch.no_grad():
        NUMCLASS  = cfg.NUM_CLASSES
        interp    = nn.Upsample(size=(256, 256), mode='bilinear',align_corners=True)
        cla_feas_src,pred_src_aux, pred_src_main = model(images_sval.cuda())

        pred_src_main = interp(pred_src_main)
        _,sval_dice_arr,sval_class_number  = dice_eval(pred=pred_src_main,label=labels_sval.cuda(),n_class=NUMCLASS)
        sval_dice_arr    = np.hstack(sval_dice_arr)
        cla_feas_trg,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        pred_trg_main  = interp(pred_trg_main)
        _,trg_dice_arr,trg_class_number = dice_eval(pred=pred_trg_main,label=labels_target.cuda(),n_class=NUMCLASS)
        trg_dice_arr   = np.hstack(trg_dice_arr)

        print('Dice')
        print('######## Source Validation Set ##########')
        print('Each Class Number {}'.format(sval_class_number))
        print('Myo:{:.3f}'.format(sval_dice_arr[1]))
        print('LAC:{:.3f}'.format(sval_dice_arr[2]))
        print('LVC:{:.3f}'.format(sval_dice_arr[3]))
        print('AA:{:.3f}'.format(sval_dice_arr[4]))
        print('######## Source Validation Set ##########')

        print('######## Target Train Set ##########')
        print('Each Class Number {}'.format(trg_class_number))
        print('Myo:{:.3f}'.format(trg_dice_arr[1]))
        print('LAC:{:.3f}'.format(trg_dice_arr[2]))
        print('LVC:{:.3f}'.format(trg_dice_arr[3]))
        print('AA:{:.3f}'.format(trg_dice_arr[4]))
        print('######## Target Train Set ##########')

def label_downsample(labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=[fea_h,fea_w], mode='nearest')
    # labels = F.interpolate(labels, size=fea_w, mode='nearest')
    # labels = labels.permute(0, 2, 1).contiguous()
    # labels = F.interpolate(labels, size=fea_h, mode='nearest')
    # labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

def update_class_center_iter(cla_src_feas,batch_src_labels,class_center_feas,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    # batch_src_feas     = cla_src_feas.detach()

    batch_src_feas = cla_src_feas
    batch_src_labels   = batch_src_labels.cuda()
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels.unsqueeze(1), fea_h, fea_w)  # n*fea_h*fea_w
    # batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w

    batch_class_center_fea_list = []
    for i in range(5):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas_update = m * class_center_feas.detach() + (1-m) * batch_class_center_feas

    return class_center_feas_update


def train_senery(model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''
    # create the model and start the training
    mseloss = torch.nn.MSELoss()
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE

    num_classes       = cfg.NUM_CLASSES
    viz_tensorboard   = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    # compute class center

    class_center_feas_src = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT_SRC).squeeze()
    class_center_feas_src = torch.from_numpy(class_center_feas_src).float().cuda()
    class_center_feas_trg = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT_TRG).squeeze()
    class_center_feas_trg = torch.from_numpy(class_center_feas_trg).float().cuda()
    # interpolate output segmaps
    # if viz_tensorboard:
    writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    #SEGMENTATION
    model.train()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    # DISCRIMINATOR NETWORK
    # feature-level
    if cfg.TRAIN.D_NORM == 'Batch_Norm':
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.InstanceNorm2d

    d_aux  = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')
    d_main = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')
    d_aux.train()
    d_aux.cuda()

    #output level
    d_main.train()
    d_main.cuda()
    print('finish model setup')
    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps

    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)
    # labels for adversarial learning

    source_label = 0
    targte_label = 1
    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN

    best_mean_dice = 0
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):
        # set train mode for each net
        model.train()
        d_main.train()
        d_aux.train()
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LE if needed
        adjust_learning_rate(optimizer,i_iter,cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux,i_iter,cfg)
        adjust_learning_rate_discriminator(optimizer_d_main,i_iter,cfg)

        #UDA training
        # First only train segmentation network based on source label
        # set discriminator require_grad = False
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source metadata

        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        cla_feas_src,pred_src_aux, pred_src_main = model(images_source.cuda())
        # adversarial training to fool the discriminator
        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target,labels_target,images_target_list, _ = batch
        cla_feas_trg, pred_trg_aux, pred_trg_main    = model(images_target.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source,cfg)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source)
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0
        pred_src_main     = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source,cfg)
        loss_dice_main    = dice_loss(pred_src_main,labels_source)
        loss_seg_all = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux)
        # adversarial training to fool the discriminator
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux     = interp(pred_trg_aux)
            d_out_aux        = d_aux(prob_2_entropy(F.softmax(pred_trg_aux,dim=1)))
            loss_adv_trg_aux = bce_loss(d_out_aux,source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main     = interp(pred_trg_main)
        d_out_main        = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
        loss_adv_trg_main = bce_loss(d_out_main,source_label)
        loss_adv_all = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main+
                cfg.TRAIN.LAMBDA_ADV_AUX  * loss_adv_trg_aux)

        ###################################################################################################
        # # if i_iter > cfg.TRAIN.warmup_epoch:
        class_center_feas_src = update_class_center_iter(cla_feas_src, labels_source, class_center_feas_src,
                                                     m=cfg.TRAIN.CLASS_CENTER_M)
        pesudo_labels_target = torch.max(F.softmax(pred_trg_main,dim=1),dim=1)[1].detach()
        class_center_feas_trg = update_class_center_iter(cla_feas_trg, pesudo_labels_target, class_center_feas_trg,
                                                     m=cfg.TRAIN.CLASS_CENTER_M)
        class_loss = 0
        for i in range(5):
            class_loss += mseloss(class_center_feas_src[i, :],class_center_feas_trg[i, :])
        # hard_pixel_label, pixel_mask = generate_pseudo_label(cla_feas_trg, class_center_feas_trg, cfg)
        # pesudo_labels_target_down = label_downsample(pesudo_labels_target.unsqueeze(1), 33,33)
        # batch_class_center_fea_list = []
        # # print(cla_feas_trg.shape)
        # for i in range(5):
        #     fea_mask = torch.eq(pesudo_labels_target_down, i).float().cuda()  # n*1*fea_h*fea_w
        #     class_feas = cla_feas_trg * fea_mask  # n*c*fea_h*fea_w
        #     # print(class_feas.shape)
        #     class_fea_sum = torch.sum(class_feas, [0, 2, 3])  # c
        #     class_num = torch.sum(fea_mask, [0, 1, 2, 3])
        #     if class_num != 0:
        #     #     batch_class_center_fea = torch.zeros([2048]).float().cuda().detach()
        #     # else:
        #         batch_class_center_fea = class_fea_sum / class_num
        #         batch_class_center_fea = batch_class_center_fea.unsqueeze(0)  # 1 * c
        #         batch_class_center_fea_list.append(batch_class_center_fea)
        # batch_class_center_feas = torch.cat(batch_class_center_fea_list, dim=0)  # n_class * c

        # cla_feas_trg_de = cla_feas_trg.clone()
        # batch, N_fea, H, W = cla_feas_trg_de.size()
        # cla_feas_trg_de = F.normalize(cla_feas_trg_de, p=2, dim=1)
        # class_centers_norm = F.normalize(batch_class_center_feas.detach(), p=2, dim=1)
        # cla_feas_trg_de = cla_feas_trg_de.permute(0, 2, 3,1).contiguous()  # N*H*W*N_fea
        # cla_feas_trg_de = torch.reshape(cla_feas_trg_de, [-1, N_fea])
        # class_centers_norm = class_centers_norm.transpose(0, 1)  # N_fea*C
        # batch_pixel_cosine = torch.matmul(cla_feas_trg_de, class_centers_norm)  # N*N_class
        # threshold = 0.05
        # batch_sort_cosine, _ = torch.sort(batch_pixel_cosine, dim=1)
        # # print(batch_sort_cosine.shape)
        # if batch_sort_cosine.shape[1]>1:
        #     pixel_sub_cosine = batch_sort_cosine[:, -1] - batch_sort_cosine[:, -2]
        #     # print(batch_sort_cosine[:, -1],batch_sort_cosine[:, -2])
        #     # pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)
        #     pixel_mask = pixel_sub_cosine > threshold
        # else:
        #     pixel_mask = batch_sort_cosine[:, -1]>0
        #     # print(pixel_mask.sum())
        # # print(pixel_mask.shape)
        # pixel_mask = pixel_mask.reshape(pred_trg_main.shape[0], 256, 256).unsqueeze(1).float()
        # pixel_mask = F.interpolate(pixel_mask.float(), size=[input_size_source[1],input_size_source[0]], mode='nearest')
        # print(pixel_mask.shape)

        # else:
        #     class_loss = 0
        ###################################################################################################

        if i_iter % 20 == 0:
            # maskp = pixel_mask.clone().cpu().data.float()[:4, ...]
            # image = maskp.repeat(1, 3, 1, 1)
            # # cor = correct_mask.clone().cpu().data
            # grid_image = make_grid(image, 4, normalize=True)
            # writer.add_image('Target/pixel_mask', grid_image, i_iter)

            grid_image = make_grid(images_target[:4, ...].clone().cpu().data, 4, normalize=True)
            writer.add_image('Target/image_origin', grid_image, i_iter)
            image = labels_target[:4, :, :]
            grid_image = make_grid(decode_seg_map_sequence(image.data.cpu().numpy()), 4, normalize=False)
            # grid_image = make_grid(color_seg(labels_target.clone()[0]).cpu().data, 1)
            writer.add_image('Target/ground_truth', grid_image, i_iter)
            # prediction_trg = pred_trg_main.clone().max(dim=1)[1]
            image = torch.max(pred_trg_main[:4, :, :, :], 1)[1].data.cpu().numpy()
            image = decode_seg_map_sequence(image)
            grid_image = make_grid(image, 4, normalize=False)
            # grid_image = make_grid(color_seg(prediction_trg[0]).cpu().data, 1)
            writer.add_image('Target/pred_trg_main', grid_image, i_iter)

            # image = pesudo_labels_target_down.squeeze().data.cpu().numpy()
            # image = decode_seg_map_sequence(image)
            # grid_image = make_grid(image, 4, normalize=False)
            # # grid_image = make_grid(color_seg(prediction_trg[0]).cpu().data, 1)
            # writer.add_image('Target/pred_trg_main_down', grid_image, i_iter)
        ###########################################################################
        # fourier_augmentation(images_source, images_target, "AS", 0.3)
        fourier_aug_list = []
        for idx in range(5):
            fourier_aug_target = torch.zeros_like(images_target)
            for img_id in range(images_target.shape[0]):
                index_src = random.randint(0, images_source.shape[0]-1)
                # L_index = random.random()*0.5
                images_source_one = np.transpose(images_source[index_src].data.numpy(), (1, 2, 0))
                images_target_one = np.transpose(images_target[img_id].data.numpy(), (1, 2, 0))
                # aug_images_target = match_histograms(images_target_one,images_source_one)
                _, aug_images_target = fourier_transform(images_source_one, images_target_one, L=0.01, i=1)
                aug_images_target = np.transpose(aug_images_target, (2, 0, 1))
                fourier_aug_target[img_id,...] = torch.from_numpy(aug_images_target).float()
            fourier_aug_list.append(fourier_aug_target)
        ###########################################################################
        pred_trg_main_label = pred_trg_main.max(dim=1)[1].reshape(-1) ### B*C*H*W==>BHW4*256*256
        bincounts = torch.bincount(pred_trg_main_label.long(), minlength=5).float() / pred_trg_main_label.size(0)

        log_q = torch.log(bincounts + 1e-12).detach()
        loss_infoent = 0.1 * torch.mean(
            torch.sum(pred_trg_main.softmax(dim=1) *
                      log_q.reshape(1,5,1,1).repeat(pred_trg_main.shape[0],1,pred_trg_main.shape[2],pred_trg_main.shape[3]), dim=1))
        _, _, score_t_og_ori = model(images_target.cuda())
        score_t_og_ori = interp(score_t_og_ori)
        score_t_og = score_t_og_ori.detach()
        tgt_preds = score_t_og.max(dim=1)[1] ###B*H*W

        #############################################for foriner pertrub############################
        fourier_correct_mask, fourier_incorrect_mask = torch.zeros_like(tgt_preds).cuda(), \
                                       torch.zeros_like(tgt_preds).cuda()  ###B*H*W

        for fourier_data_t_aug_curr in fourier_aug_list:
            _, _, fourier_score_t_aug_curr = model(fourier_data_t_aug_curr.cuda())
            fourier_score_t_aug_curr = interp(fourier_score_t_aug_curr)
            # score_t_aug_curr = score_t_aug_curr.reshape(-1,score_t_aug_curr.shape[1])
            fourier_tgt_preds_aug = fourier_score_t_aug_curr.max(dim=1)[1] ###B*H*W
            fourier_consistent_idxs = (tgt_preds == fourier_tgt_preds_aug).detach() ###B*H*W
            fourier_inconsistent_idxs = (tgt_preds != fourier_tgt_preds_aug).detach() ###B*H*W
            fourier_correct_mask = fourier_correct_mask + fourier_consistent_idxs.type(torch.uint8) ###B*H*W
            fourier_incorrect_mask = fourier_incorrect_mask + fourier_inconsistent_idxs.type(torch.uint8) ###B*H*W

            # fourier_consistent_idxs_x = fourier_consistent_idxs.unsqueeze(1).repeat(1,5,1,1) ###B*C*H*W
            # fourier_inconsistent_idxs_x = fourier_inconsistent_idxs.unsqueeze(1).repeat(1,5,1,1) ###B*C*H*W
        fourier_correct_mask, fourier_incorrect_mask = fourier_correct_mask >= 3, fourier_incorrect_mask >= 3
        torch.cuda.empty_cache()
        #############################################for foriner pertrub############################
        # Compute actual correctness mask for analysis only
        correct_mask_gt = (tgt_preds.detach().cpu() == labels_target)

        correct_mask, incorrect_mask = torch.zeros_like(tgt_preds).cuda(), \
                                       torch.zeros_like(tgt_preds).cuda()  ###B*H*W

        score_t_aug_pos, score_t_aug_neg = torch.zeros_like(score_t_og), \
                                           torch.zeros_like(score_t_og) ###B*C*H*W
        for data_t_aug_curr in images_target_list:
            # noise = torch.randn_like(data_t_aug_curr) * 3
            # # print(noise.max(),noise.min())
            # # print(data_t_aug_curr.max(),data_t_aug_curr.min())
            # data_t_aug_curr = data_t_aug_curr + noise
            _, _, score_t_aug_curr = model(data_t_aug_curr.cuda())
            score_t_aug_curr = interp(score_t_aug_curr)
            # score_t_aug_curr = score_t_aug_curr.reshape(-1,score_t_aug_curr.shape[1])
            tgt_preds_aug = score_t_aug_curr.max(dim=1)[1] ###B*H*W
            consistent_idxs = (tgt_preds == tgt_preds_aug).detach() ###B*H*W
            inconsistent_idxs = (tgt_preds != tgt_preds_aug).detach() ###B*H*W
            correct_mask = correct_mask + consistent_idxs.type(torch.uint8) ###B*H*W
            incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8) ###B*H*W

            consistent_idxs_x = consistent_idxs.unsqueeze(1).repeat(1,5,1,1) ###B*C*H*W
            inconsistent_idxs_x = inconsistent_idxs.unsqueeze(1).repeat(1,5,1,1) ###B*C*H*W

            score_t_aug_pos = torch.where(consistent_idxs_x, score_t_aug_curr, score_t_aug_pos) ###B*C*H*W
            score_t_aug_neg = torch.where(inconsistent_idxs_x, score_t_aug_curr, score_t_aug_neg) ####B*C*H*W
        # print(correct_mask.max(),correct_mask.min(),correct_mask.shape)
        # print(incorrect_mask.max(), incorrect_mask.min(), incorrect_mask.shape)
        correct_mask, incorrect_mask = correct_mask >= 3, incorrect_mask >= 3
        torch.cuda.empty_cache()   ################################清除缓存
        # print(correct_mask.shape)
        if i_iter % 20 == 0:
            grid_image = make_grid(fourier_data_t_aug_curr[:4, ...].clone().cpu().data, 4, normalize=True)
            writer.add_image('Target/fourier_image_augment', grid_image, i_iter)

            grid_image = make_grid(data_t_aug_curr[:4, ...].clone().cpu().data, 4, normalize=True)
            writer.add_image('Target/image_augment', grid_image, i_iter)
            # # prediction_trg_aug = score_t_aug_curr.clone().max(dim=1)[1]
            # image = torch.max(score_t_aug_pos[:4, :, :, :], 1)[1].data.cpu().numpy()
            # image = decode_seg_map_sequence(image)
            # grid_image = make_grid(image, 4, normalize=False)
            # # grid_image = make_grid(color_seg(prediction_trg_aug[0]).cpu().data, 1)
            # writer.add_image('Target/pred_trg_aug_pos', grid_image, i_iter)
            #
            # image = torch.max(score_t_aug_neg[:4, :, :, :], 1)[1].data.cpu().numpy()
            # image = decode_seg_map_sequence(image)
            # grid_image = make_grid(image, 4, normalize=False)
            # # grid_image = make_grid(color_seg(prediction_trg_aug[0]).cpu().data, 1)
            # writer.add_image('Target/pred_trg_aug_neg', grid_image, i_iter)

            maska = correct_mask_gt.clone().cpu().data.float()[:4, ...]
            image = maska.unsqueeze(1).repeat(1, 3, 1, 1)
            # cor = correct_mask.clone().cpu().data
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/correct_mask_gt', grid_image, i_iter)

            mask2 = correct_mask.clone().cpu().data.float()[:4, ...]
            image = mask2.unsqueeze(1).repeat(1, 3, 1, 1)
            # cor = correct_mask.clone().cpu().data
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/correct_mask', grid_image, i_iter)
            mask = incorrect_mask.clone().cpu().data.float()[:4, ...]
            image = mask.unsqueeze(1).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/incorrect_mask', grid_image, i_iter)

            mask2 = fourier_correct_mask.clone().cpu().data.float()[:4, ...]
            image = mask2.unsqueeze(1).repeat(1, 3, 1, 1)
            # cor = correct_mask.clone().cpu().data
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/fourier_correct_mask', grid_image, i_iter)
            mask = fourier_incorrect_mask.clone().cpu().data.float()[:4, ...]
            image = mask.unsqueeze(1).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/fourier_incorrect_mask', grid_image, i_iter)
        # Compute some stats
        # if batch_sort_cosine.shape[1] > 1:
        correct_mask=(correct_mask & fourier_correct_mask)
        incorrect_mask=(incorrect_mask | fourier_incorrect_mask)
        # .float().cuda()).squeeze().bool()
        if i_iter % 20 == 0:
            mask2 = correct_mask.clone().cpu().data.float()[:4, ...]
            image = mask2.unsqueeze(1).repeat(1, 3, 1, 1)
            # cor = correct_mask.clone().cpu().data
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/correct_mask_after', grid_image, i_iter)
            mask = incorrect_mask.clone().cpu().data.float()[:4, ...]
            image = mask.unsqueeze(1).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 4, normalize=True)
            writer.add_image('Target/incorrect_mask_after', grid_image, i_iter)
        correct_ratio = (correct_mask).sum().item() / \
                        (images_target.shape[0]*images_target.shape[2]*images_target.shape[3])
        incorrect_ratio = (incorrect_mask).sum().item() / \
                          (images_target.shape[0]*images_target.shape[2]*images_target.shape[3])

        # consistency_conf_mat, correct_precision, correct_recall, correct_f1 = compute_prf1(
        #     correct_mask_gt.clone().numpy(), \
        #     correct_mask.clone().cpu().numpy())
        # print("\n {:d} / {:d} consistent ({:.2f}): GT precision: {:.2f}: GT recall: {:.2f}".format(correct_mask.sum(),
        # images_target.shape[0]*images_target.shape[2]*images_target.shape[3], correct_ratio, correct_precision, correct_recall))

        writer.add_scalar('ratios/correct_ratio', correct_ratio, i_iter)
        writer.add_scalar('ratios/incorrect_ratio', incorrect_ratio, i_iter)
        # correct_mask, incorrect_mask = correct_mask.reshape(-1), incorrect_mask.reshape(-1) ##BHW

        if correct_ratio > 0.0:
            probs_t_pos = F.softmax(score_t_aug_pos, dim=1)
            probs_t_pos_sum = torch.sum(probs_t_pos* (torch.log(probs_t_pos + 1e-12)), 1)
            # print(torch.masked_select(probs_t_pos_sum, correct_mask).shape)
            loss_cent_correct = 1 * correct_ratio * -torch.mean(torch.masked_select(probs_t_pos_sum, correct_mask))
        else:
            loss_cent_correct=0

        if incorrect_ratio > 0.0:
            probs_t_neg = F.softmax(score_t_aug_neg, dim=1)
            probs_t_neg_sum = torch.sum(probs_t_neg * (torch.log(probs_t_neg + 1e-12)), 1)
            loss_cent_incorrect = 1 * incorrect_ratio * torch.mean(torch.masked_select(probs_t_neg_sum, incorrect_mask))
            # print(torch.masked_select(probs_t_neg_sum, incorrect_mask).shape)
        else:
            loss_cent_incorrect=0

        ###########################################################################
        loss_total = loss_seg_all+loss_adv_all+loss_cent_correct+loss_cent_incorrect+class_loss
        # +loss_infoent+class_loss
        loss_total.backward()

        #Train discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        #First we train d with source metadata
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux    = d_aux(prob_2_entropy(F.softmax(pred_src_aux,dim=1)))
            loss_d_aux   = bce_loss(d_out_aux,source_label)
            loss_d_aux   = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main    = d_main(prob_2_entropy(F.softmax(pred_src_main,dim=1)))
        loss_d_main   = bce_loss(d_out_main,source_label)
        loss_d_main   = loss_d_main / 2
        loss_d_main.backward()

        # second we train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux    = d_aux(prob_2_entropy(F.softmax(pred_trg_aux,dim=1)))
            loss_d_aux   = bce_loss(d_out_aux,targte_label)
            loss_d_aux   = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main  = pred_trg_main.detach()
        d_out_main     = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
        loss_d_main    = bce_loss(d_out_main,targte_label)
        loss_d_main    = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()
        ####################################
        torch.cuda.empty_cache()
        ####################################
        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_adv_trg_aux' :loss_adv_trg_aux,
                          'loss_adv_trg_main':loss_adv_trg_main,
                          'loss_d_aux'       :loss_d_aux,
                          'loss_d_main'      :loss_d_main,
                          "loss_cent_correct":loss_cent_correct,
                          "loss_cent_incorrect":loss_cent_incorrect,
                          "loss_infoent":loss_infoent,
                          'loss_class'       :class_loss}
        if i_iter % 200 == 0 and i_iter != 0:
            print_losses(current_losses,i_iter)

            # try:
            #     _, batch = sval_loader_iter.__next__()
            # except StopIteration:
            #     sval_loader_iter = enumerate(sval_loader)
            #     _, batch = sval_loader_iter.__next__()
            # images_sval, labels_sval,_ = batch
            #
            # iter_eval(model, images_sval, labels_sval, images_target, labels_target, cfg)
        if i_iter % 500 == 0 and i_iter != 0:
            if cfg.TARGET == 'CT':
                test_list_pth = '../data/datalist/test_ct.txt'
            if cfg.TARGET == 'MR':
                test_list_pth = '../data/datalist/test_mr.txt'
            with open(test_list_pth) as fp:
                rows = fp.readlines()
            testfile_list = [row[:-1] for row in rows]
            dice_mean, dice_std, assd_mean, assd_std = eval_during_train(testfile_list, model, cfg.TARGET)
            is_best = np.mean(dice_mean) > best_mean_dice
            best_mean_dice = max(np.mean(dice_mean),best_mean_dice)
            if is_best:
                print('Dice:')
                print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
                print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
                print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
                print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
                print('Mean:%.1f' % np.mean(dice_mean))
                print('ASSD:')
                print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
                print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
                print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
                print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
                print('Mean:%.1f' % np.mean(assd_mean))
                print('taking snapshot ...')
                print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
                snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
                torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
                torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
                torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')

            model.train()
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer,current_losses,i_iter)
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cal_variance(pred, aug_pred):
    kl_distance = nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    variance = torch.sum(kl_distance(
        log_sm(pred), sm(aug_pred)), dim=1)
    exp_variance = torch.exp(-variance)
    return variance, exp_variance
def get_current_consistency_weight(epoch,cfg):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return cfg.TRAIN.consistency * sigmoid_rampup(epoch, cfg.TRAIN.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class CriterionMiniBatchCrossImagePair(nn.Module):
    def __init__(self, temperature):
        super(CriterionMiniBatchCrossImagePair, self).__init__()
        self.temperature = temperature

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def forward(self, feat_S, feat_T):
        # feat_T = self.concat_all_gather(feat_T)
        # feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()
        # print(feat_S.size())

        '''
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        '''
        patch_w = 4
        patch_h = 4
        # maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        sim_dis = torch.tensor(0.).cuda()
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])
                # print(s_sim_map.shape)

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return sim_dis

def log_losses_tensorboard(writer,current_losses,i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value),i_iter)

def print_losses(current_losses,i_iter):
    list_strings = []
    for loss_name,loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.3f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor,(int,float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg):

    if cfg.TRAIN.DA_METHOD == 'MPSCL':
        print("use method:",cfg.TRAIN.DA_METHOD)
        train_mpscl(model, strain_loader, sval_loader,trgtrain_loader, cfg)
    if cfg.TRAIN.DA_METHOD == 'AdvEnt':
        print("use method:", cfg.TRAIN.DA_METHOD)
        train_advent(model, strain_loader, sval_loader,trgtrain_loader, cfg)
    if cfg.TRAIN.DA_METHOD == 'Adaoutput':
        print("use method:", cfg.TRAIN.DA_METHOD)
        train_adaoutput(model, strain_loader, sval_loader,trgtrain_loader, cfg)
    if cfg.TRAIN.DA_METHOD == "senery":
        print("use method:", cfg.TRAIN.DA_METHOD)
        train_senery(model, strain_loader, sval_loader,trgtrain_loader, cfg)

def train_domain_adaptation_MT(model,ema_model,strain_loader,trgtrain_loader,sval_loader,cfg):
    if cfg.TRAIN.DA_METHOD == "MT":
        print("use method:", cfg.TRAIN.DA_METHOD)
        train_unbiased_mt(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
def load_checkpoint(model, checkpoint,):
    saved_state_dict = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)

