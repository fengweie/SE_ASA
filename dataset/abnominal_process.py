import os
import shutil
from time import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from skimage.io import imread, imsave
from skimage import transform
import matplotlib.pyplot as plt # plt 用于显示图片
# import dicom
import glob
from PIL import Image
dcm_dir = sorted(glob.glob('/mnt/workdir/fengwei/ultra_wide/DA_vessel/CHAOS_Train_Sets/Train_Sets/MR/*/T2SPIR/DICOM_anon/'))#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/img/'
seg_dir = sorted(glob.glob('/mnt/workdir/fengwei/ultra_wide/DA_vessel/CHAOS_Train_Sets/Train_Sets/MR/*/T2SPIR/Ground/'))#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/label/'
new_ct_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/CHAOS_Train_Sets/image_process/'
new_seg_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/CHAOS_Train_Sets/label_process/'

upper = 350#multi-A#200
lower = -upper#-200
expand_slice = 10  # 轴向上向外扩张的slice数量
size = 48  # 取样的slice数量
stride = 3  # 取样的步长
down_scale = 0.5
slice_thickness = 100#3#33
# test_ct_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_image/'
# test_seg_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_mask/'
# os.mkdir(new_ct_dir)
# os.mkdir(new_seg_dir)
# # os.mkdir(test_ct_dir)
if not os.path.exists(new_ct_dir):
    os.mkdir(new_ct_dir)
    os.mkdir(new_seg_dir)
# os.mkdir(new_ct_dir)
# os.mkdir(new_seg_dir)
file_index = 0
test_index = 0
for dcm_file in dcm_dir:

    # 将CT和金标准入读内存
    # dcm = dicom.read_file(os.path.join(dcm_dir, dcm_file))
    ct = sitk.ReadImage(dcm_file, sitk.sitkInt16)
    # ct = Image.open(os.path.join(dcm_dir, dcm_file))
    # ct.SetDirection(ct.GetDirection())
    # ct.SetOrigin(ct.GetOrigin())
    # ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))
    # seg = Image.open(os.path.join(seg_dir, dcm_file.replace('DICOM_anon', 'Ground').replace('dcm', 'png')))
    # seg_array = np.array(seg)
    # plt.imshow(seg_array)
    seg = sitk.ReadImage(dcm_file.replace('DICOM_anon', 'Ground').replace('dcm', 'png'), sitk.sitkInt16)
    # seg.SetDirection(ct.GetDirection())
    # seg.SetOrigin(ct.GetOrigin())
    # seg.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

    ct_name=dcm_file.replace('.dcm', '')
    seg_name = ct_name#.replace('', 'label')
    # print(dcm_file)
    # seg = ct.resize(256,256)

    ct_array = sitk.GetArrayFromImage(ct)
    print(ct_array.shape)


    seg_array = sitk.GetArrayFromImage(seg)
    seg_array1 = sitk.GetArrayFromImage(seg)
    seg_array2 = sitk.GetArrayFromImage(seg)
    seg_array3 = sitk.GetArrayFromImage(seg)
    seg_array4 = sitk.GetArrayFromImage(seg)

    # if seg_array.shape[0]>260:
    #     seg_array=seg_array[32:288,32:288]
    #     seg_array1=seg_array1[32:288,32:288]
    #     seg_array2=seg_array2[32:288,32:288]
    #     seg_array3=seg_array3[32:288,32:288]
    #     seg_array4=seg_array4[32:288,32:288]
    #     ct_array = ct_array[:,32:288,32:288]

    print(seg_array.shape)
    # plt.subplot(121)
    # plt.imshow(ct_array[0,:,:])
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(seg_array)#, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()
    # ct_array[ct_array > upper] = upper
    # ct_array[ct_array < lower] = lower
    # Len = len(ct_array)
    # print(Len)
    # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    # print(ct_array)
    # ct_array2 = ndimage.zoom(ct_array2, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    # seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=0)
    # for i in range(2,len(ct_array)):
    new_ct_array = ct_array[0, :, :]
    new_seg_array = seg_array#[0, :, :]

        # new_ct = np.array(new_ct_array,dtype='uint8')
        # new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))
        # t=predict[0,0,:,:]
        # # print(t.shape)
        # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
        # # print(tt.shape)   #(256.256)
        # # tt=np.zeros((256,256,3))
        # # tt[:,:,0]=t[0,:,:]
        # # # tt[:,:,1]=t[0,:,:]
        # # # tt[:,:,2]=t[1,:,:]
        # # print(tt[0,100,100])
        # # m=target[1,:,:,:]
        # # mm=np.zeros((256,256))
        # mm=target[0,:,:]
        ##print(np.maximum(mm, -1))
        # # mm[:,:,1]=target[1,:,:]
        # # mm[:,:,2]=target[2,:,:]
        # # t = t.transpose((1,2,0))


        # new_seg = np.array(new_seg_array,dtype='uint8')
        # print(new_seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))
    count_0=sum(new_seg_array!=0)    #剔除空白分割图

        # count_s = sum(new_seg_array & seg_array[i-1, :, :]) #剔除相近的图
    # print(sum(count_0)) #1 4
    # print(np.min(new_seg_array),np.max(new_seg_array))
        # if sum(count0)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01 and sum(count_s)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01:
    if sum(count_0)>0:
        # plt.subplot(121)
        # plt.imshow(new_ct_array)
        # # plt.show()
        # plt.subplot(122)
        # plt.imshow(new_seg_array)#, cmap='Greys_r') # 显示图片
        # # plt.axis('off') # 不显示坐标轴
        # plt.show()
        # print(count0)
        # file_index += 1
        # print(new_ct_array.shape,new_seg_array.shape)
        new_ct_name =  ct_name + '.png'
        new_seg_name = seg_name+ '.png'
        # new_seg_array = transform.resize(new_seg_array,(256,256))
        # print(new_ct_name)
        # print(new_seg_name)
        # test_index += 1
        # if test_index ==100:
        #     imsave(test_seg_dir+new_seg_name, new_seg_array)
        #     imsave(test_ct_dir+new_ct_name, new_ct_array)
        #     test_index=0
        # else:
        # print(new_seg_dir+new_seg_name,new_ct_array.shape)
        rightKidney_label = seg_array1
        rightKidney_label[rightKidney_label!=120]=0
        rightKidney_label[rightKidney_label==120]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'rightKidney_label')):
            os.mkdir(os.path.join(new_seg_dir, 'rightKidney_label'))
        imsave(new_seg_dir+'rightKidney_label/'+new_seg_name, rightKidney_label)

        liver_label = seg_array2
        liver_label[liver_label!=63]=0
        liver_label[liver_label==63]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'liver_label')):
            os.mkdir(os.path.join(new_seg_dir, 'liver_label'))
        imsave(new_seg_dir+'liver_label/'+new_seg_name, liver_label)

        leftKidney_label = seg_array3
        leftKidney_label[leftKidney_label!=189]=0
        leftKidney_label[leftKidney_label==189]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'leftKidney_label')):
            os.mkdir(os.path.join(new_seg_dir, 'leftKidney_label'))
        imsave(new_seg_dir+'leftKidney_label/'+new_seg_name, leftKidney_label)
        spleen_label = seg_array4
        spleen_label[spleen_label!=252]=0
        spleen_label[spleen_label==252]=255
        if not os.path.exists(os.path.join(new_seg_dir, 'spleen_label')):
            os.mkdir(os.path.join(new_seg_dir, 'spleen_label'))
        imsave(new_seg_dir+'spleen_label/'+new_seg_name, spleen_label)

        imsave(new_seg_dir+new_seg_name, new_seg_array)
        imsave(new_ct_dir+new_ct_name, new_ct_array)
    file_index = 0
        # cv2.imwrite(os.path.join(new_seg_dir, new_seg_name),new_seg)
        # plt.savefig(new_seg, os.path.join(new_seg_dir, new_seg_name))
        # sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
        # sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))
    # print(ct_array.shape)



    # print(seg_array.shape)

# import os
# import shutil
# from time import time
# import numpy as np
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import scipy.ndimage as ndimage
# from skimage.io import imread, imsave
# from skimage import transform
# import matplotlib.pyplot as plt # plt 用于显示图片
# ct_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/RawData/RawData/Training/img/'#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/img/'
# seg_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/RawData/RawData/Training/label/'#'/home/lc/Study/DataBase/multi-Atlas labeling beyond the cranial vault/Training/label/'
# new_ct_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/RawData/image_process/'
# new_seg_dir = '/mnt/workdir/fengwei/ultra_wide/DA_vessel/RawData/label_process/'
#
# upper = 350#multi-A#200
# lower = -upper#-200
# expand_slice = 10  # 轴向上向外扩张的slice数量
# size = 48  # 取样的slice数量
# stride = 3  # 取样的步长
# down_scale = 0.5
# slice_thickness = 10#100#3#33
# # test_ct_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_image/'
# # test_seg_dir = '/home/lc/学习/DataBase/LITS(1)/png/test_mask/'
# # os.mkdir(new_ct_dir)
# # os.mkdir(new_seg_dir)
# # # os.mkdir(test_ct_dir)
# os.mkdir(new_ct_dir)
# os.mkdir(new_seg_dir)
# file_index = 0
# test_index = 0
# for ct_file in os.listdir(ct_dir):
#
#     # 将CT和金标准入读内存
#
#     ct = sitk.ReadImage(os.path.join(ct_dir, ct_file), sitk.sitkInt16)
#     # ct.SetDirection(ct.GetDirection())
#     # ct.SetOrigin(ct.GetOrigin())
#     # ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))
#
#     seg = sitk.ReadImage(os.path.join(seg_dir, ct_file.replace('img', 'label')), sitk.sitkInt16)
#     # seg.SetDirection(ct.GetDirection())
#     # seg.SetOrigin(ct.GetOrigin())
#     # seg.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))
#
#     ct_name=ct_file.replace('.nii.gz', '_')
#     seg_name = ct_name#.replace('image', 'label')
#     print(ct_file)
#     # seg = ct.resize(256,256)
#
#     ct_array = sitk.GetArrayFromImage(ct)#.resize([256,256])
#     seg_array = sitk.GetArrayFromImage(seg)
#     seg_array1 = sitk.GetArrayFromImage(seg)
#     seg_array2 = sitk.GetArrayFromImage(seg)
#     seg_array3 = sitk.GetArrayFromImage(seg)
#     seg_array4 = sitk.GetArrayFromImage(seg)
#
#
#     ct_array[ct_array > upper] = upper
#     ct_array[ct_array < lower] = lower
#     Len = len(ct_array)
#     # print(Len)
#     # print(ct_array.shape)
#     ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[0], down_scale, down_scale), order=3)
#     #
#     # print('shape is :',ct_array.shape)
#     seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[0], down_scale, down_scale), order=0)
#     seg_array1 = ndimage.zoom(seg_array1, (ct.GetSpacing()[0], down_scale, down_scale), order=0)
#     seg_array2 = ndimage.zoom(seg_array2, (ct.GetSpacing()[0], down_scale, down_scale), order=0)
#     seg_array3 = ndimage.zoom(seg_array3, (ct.GetSpacing()[0], down_scale, down_scale), order=0)
#     seg_array4 = ndimage.zoom(seg_array4, (ct.GetSpacing()[0], down_scale, down_scale), order=0)
#     # print('seg_array shape is :',seg_array.shape)
#     seg_array[seg_array==4]=0
#     seg_array[seg_array==5]=0
#     seg_array[seg_array==7]=0
#     seg_array[seg_array==8]=0
#     seg_array[seg_array==9]=0
#     seg_array[seg_array==10]=0
#     seg_array[seg_array==11]=0
#     seg_array[seg_array==12]=0
#     seg_array[seg_array==13]=0
#     for i in range(2,len(ct_array)):
#         # print('i,',i)
#         #
#         new_ct_array = ct_array[i, :, :]#.data.resize([256,256])
#         new_seg_array = seg_array[i, :, :]
#         new_seg_array1 = seg_array1[i, :, :]
#         new_seg_array2 = seg_array2[i, :, :]
#         new_seg_array3 = seg_array3[i, :, :]
#         new_seg_array4 = seg_array4[i, :, :]
#         new_ct_array=np.flipud(new_ct_array)
#         new_seg_array=np.flipud(new_seg_array)
#         new_seg_array1=np.flipud(new_seg_array1)
#         new_seg_array2=np.flipud(new_seg_array2)
#         new_seg_array3=np.flipud(new_seg_array3)
#         new_seg_array4=np.flipud(new_seg_array4)
#         # print(new_ct_array.shape)
#         # new_seg_array = seg_array[i, :, :]
#         # new_ct = np.array(new_ct_array,dtype='uint8')
#         # new_ct.SetOrigin(ct.GetOrigin())
#         # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))
#         # t=predict[0,0,:,:]
#         # # print(t.shape)
#         # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
#         # # print(tt.shape)   #(256.256)
#         # # tt=np.zeros((256,256,3))
#         # # tt[:,:,0]=t[0,:,:]
#         # # # tt[:,:,1]=t[0,:,:]
#         # # # tt[:,:,2]=t[1,:,:]
#         # # print(tt[0,100,100])
#         # # m=target[1,:,:,:]
#         # # mm=np.zeros((256,256))
#         # mm=target[0,:,:]
#         ##print(np.maximum(mm, -1))
#         # # mm[:,:,1]=target[1,:,:]
#         # # mm[:,:,2]=target[2,:,:]
#         # # t = t.transpose((1,2,0))
#
#
#         # new_seg = np.array(new_seg_array,dtype='uint8')
#         # print(new_seg_array)
#         # new_seg.SetDirection(ct.GetDirection())
#         # new_seg.SetOrigin(ct.GetOrigin())
#         # new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))
#         count0=sum(new_seg_array>0)    #剔除空白分割图
#
#         count_s = sum(new_seg_array & seg_array[i-1, :, :]) #剔除相近的图
#         # print(sum(count0)) #1 4
#         # if sum(count0)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01 and sum(count_s)/(new_seg_array.shape[0]*new_seg_array.shape[1])>0.01:
#         if sum(count0)>0:#/(new_seg_array.shape[0]*new_seg_array.shape[1])>0:
#             # plt.subplot(121)
#             # plt.imshow(new_ct_array)
#             # # plt.show()
#             # plt.subplot(122)
#             # plt.imshow(new_seg_array)#, cmap='Greys_r') # 显示图片
#             # # plt.axis('off') # 不显示坐标轴
#             # plt.show()
#             # print(count0)
#             file_index += 1
#             new_ct_name =  ct_name+str(file_index) + '.png'
#             new_seg_name = seg_name+str(file_index) + '.png'
#             # new_seg_array = transform.resize(new_seg_array,(256,256))
#             # print(new_ct_name)
#             # print(new_seg_name)
#             # test_index += 1
#             # if test_index ==100:
#             #     imsave(test_seg_dir+new_seg_name, new_seg_array)
#             #     imsave(test_ct_dir+new_ct_name, new_ct_array)
#             #     test_index=0
#             # else:
#             print(new_seg_dir+new_seg_name)
#
#             rightKidney_label = new_seg_array1
#             rightKidney_label[rightKidney_label!=2]=0
#             rightKidney_label[rightKidney_label==2]=255
#             # rightKidney_label.resize(256,256)
#             if not os.path.exists(os.path.join(new_seg_dir, 'rightKidney_label')):
#                 os.mkdir(os.path.join(new_seg_dir, 'rightKidney_label'))
#             # if sum(sum(rightKidney_label>0))>0:
#             imsave(new_seg_dir+'rightKidney_label/'+new_seg_name, rightKidney_label)
#
#             liver_label = new_seg_array2
#             liver_label[liver_label!=6]=0
#             liver_label[liver_label==6]=255
#             # liver_label.resize(256,256)
#             if not os.path.exists(os.path.join(new_seg_dir, 'liver_label')):
#                 os.mkdir(os.path.join(new_seg_dir, 'liver_label'))
#             # if sum(sum(liver_label>0))>0:
#             imsave(new_seg_dir+'liver_label/'+new_seg_name, liver_label)
#
#             leftKidney_label = new_seg_array3
#             leftKidney_label[leftKidney_label!=3]=0
#             leftKidney_label[leftKidney_label==3]=255
#             # leftKidney_label.resize(256,256)
#             if not os.path.exists(os.path.join(new_seg_dir, 'leftKidney_label')):
#                 os.mkdir(os.path.join(new_seg_dir, 'leftKidney_label'))
#             # if sum(sum(leftKidney_label>0))>0:
#             imsave(new_seg_dir+'leftKidney_label/'+new_seg_name, leftKidney_label)
#
#             spleen_label = new_seg_array4
#             spleen_label[spleen_label!=1]=0
#             spleen_label[spleen_label==1]=255
#             # spleen_label.resize(256,256)
#             if not os.path.exists(os.path.join(new_seg_dir, 'spleen_label')):
#                 os.mkdir(os.path.join(new_seg_dir, 'spleen_label'))
#             # if sum(sum(spleen_label>0))>0:
#             imsave(new_seg_dir+'spleen_label/'+new_seg_name, spleen_label)
#
#             # new_seg_array.resize(256,256)
#             # new_ct_array.resize(256,256)
#             imsave(new_seg_dir+new_seg_name, new_seg_array)
#             imsave(new_ct_dir+new_ct_name, new_ct_array)
#     file_index = 0
#         # cv2.imwrite(os.path.join(new_seg_dir, new_seg_name),new_seg)
#         # plt.savefig(new_seg, os.path.join(new_seg_dir, new_seg_name))
#         # sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
#         # sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))
#     # print(ct_array.shape)
#
#
#
#     # print(seg_array.shape)
