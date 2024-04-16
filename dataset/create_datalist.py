import os



def make_datalist(data_fd, data_list,data_gt_list,gt_fd):

    filename_all = os.listdir(data_fd)
    # print(filename_all[0])
    filename_all = [data_fd + img_name + '\n' for img_name in filename_all if img_name.endswith('.npy')]
    print(filename_all[0])
    print(len(filename_all))
    gt_filename_all = []
    for one_img_name in filename_all:
        gt_id = os.path.splitext(one_img_name.split("/")[-1])[0]+ '_gt.npy'
        # print(gt_id,gt_fd+gt_id)
        gt_filename_all.append(gt_fd+gt_id+ '\n')
    print(len(gt_filename_all))
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)

    with open(data_gt_list, 'w') as fp:
        fp.writelines(gt_filename_all)




if __name__ == '__main__':

    #Plz change the path follow your setting
    data_fd      = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/val_mr/'
    gt_fd = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/gt_val_mr/'
    data_list    = '../data/datalist/val_mr.txt'
    data_gt_list = '../data/datalist/val_mr_gt.txt'
    make_datalist(data_fd, data_list,data_gt_list, gt_fd)

