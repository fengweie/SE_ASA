import os
def make_datalist(root_dir,data_list):

    data_names = os.listdir(root_dir)
    data_pth  = [data_fd + data_name + '\n' for data_name in data_names if data_name.endswith('.npz')]

    with open(data_list, 'w') as fp:
        fp.writelines(data_pth)




if __name__ == '__main__':

    #Plz change the path follow your setting
    data_fd   = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/data_np/data_np/test_mr/' # data folder'
    data_list = '../data/datalist/test_mr.txt'  # txt file pth
    make_datalist(data_fd, data_list)