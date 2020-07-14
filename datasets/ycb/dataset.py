import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
# from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        if mode == 'test':
            self.add_noise = False

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                input_line = input_line[:8] + str(input_line[-5]) + input_line[8:]
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        self.cld[0] = []
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()
            
            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.refine = refine
        self.front_num = 2

        print(len(self.list))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))

        # print("img path : " + '{0}/{1}-color.png'.format(self.root, self.list[index]))
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(depth)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        # print("labels1 type : " + str(type(label)))
        # print("labels1 shape : " + str(label.shape))
        # uint_img = np.array(label * 255).astype('uint8')
        # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('{0}/{1}-label.png'.format(self.root, self.list[index]), grayImage)
        # cv2.waitKey(0)
        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        # print("labels2 type : " + str(type(label)))
        # print("labels2 shape : " + str(label.shape))
        # uint_img = np.array(label * 255).astype('uint8')
        # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("labels_mat2", grayImage)
        # cv2.waitKey(0)

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        # img_check = np.array(img)[:, :, :3]
        # img_check.astype(np.float32)
        # image_mat0 = img_check[:, :, 0].copy()
        # img_check[:, :, 0] = img_check[:, :, 2]
        # img_check[:, :, 2] = image_mat0

        # plt.imshow(img)
        # plt.show()
        # print("img_check type : " + str(type(img_check)))
        # print("img_check shape : " + str(img_check.shape))
        # cv2.imshow("img_check", img_check)
        # cv2.waitKey(0)
        if self.add_noise:
            img = self.trancolor(img)

        # img_check = np.array(img)[:, :, :3]
        # img_check.astype(np.float32)
        # image_mat0 = img_check[:, :, 0].copy()
        # img_check[:, :, 0] = img_check[:, :, 2]
        # img_check[:, :, 2] = image_mat0

        # plt.imshow(img)
        # plt.show()
        # print("img_check type : " + str(type(img_check)))
        # print("img_check shape : " + str(img_check.shape))
        # cv2.imshow("img_check", img_check)
        # cv2.waitKey(0)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))
            # print("img type : " + str(type(img)))
            # Image.Image.
            img_masked = back * mask_back
            # print("shape : " + str(img_masked.shape))
            img_masked += img
        else:
            img_masked = img
        # print("add_front : " + str(add_front))
        if self.add_noise and add_front:
            img_masked = img_masked * mask_front + front * ~(mask_front)

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)


        # images_mat = np.transpose(img_masked, (1, 2, 0))
        # print(images_mat)
        # images_mat = images_mat.astype(np.float32)
        # image_mat0 = images_mat[:, :, 0].copy()
        # images_mat[:, :, 0] = images_mat[:, :, 2]
        # images_mat[:, :, 2] = image_mat0
        # images_mat /= 255.0
        # cv2.imshow("img_masked", images_mat)
        # cv2.waitKey(0)

        # print("images type : " + str(type(img)))
        # print("images shape : " + str(img.shape))
        # print("labels type : " + str(type(label)))
        # print("labels shape : " + str(label.shape))
        # print("img_masked type : " + str(type(img_masked)))
        # print("img_masked shape : " + str(img_masked.shape))

        # print(self.cld[21])
        label_tensor = np.zeros((len(self.cld), label.shape[0], label.shape[1]))
        # print("len : " + str(len(self.cld)))
        # print("label_tensor shape : " + str(label_tensor.shape))
        # label_tensor = label_tensor.fill(-1)
        # print("label shape : " + str(label.shape))
        for j in range(1,len(self.cld)):
            # max_val = 0
            # row, col = label.shape
            # for i in range(row):
            #     for k in range(col):
            #         if label[i][k] > max_val:
            #             max_val = label[i][k]
            # print(str(j)+"label Max value : " + str(max_val))
            # label_tensor[j, :] = (label == j+1)
            label_tensor[j, :] = (label == j)
            # label_tensor[j, :] = (label == j+1) * (j+1)

            # if j == 0:
            #     arr = []
            #     row, col = label_tensor[j].shape
            #     for i in range(row):
            #         for k in range(col):
            #             if label[i][k] == j+1:
            #                 label_tensor[j][i][k] = j+1
            #
            #     row, col = label_tensor[j].shape
            #     for i in range(row):
            #         for k in range(col):
            #             if label_tensor[j][i][k] not in arr:
            #                 arr.append(label_tensor[j][i][k])
            #     print(str(j)+"label arr : " + str(arr))
        # final_label_tensor = label_tensor[0].fill(-1)
        # final_label_tensor = np.argmax(label_tensor, axis = 0)
        label_tensor = np.argmax(label_tensor, axis = 0)
        # label_tensor -= 1
        # print(label_tensor)
        # print("label tensor shape : " + str(label_tensor.shape))
        # label_tensor = label_tensor.astype(np.uint8)

        # print(str(label.shape))

        # arr = []
        # row, col = label_tensor.shape
        # for i in range(row):
        #     for j in range(col):
        #         if label_tensor[i][j] not in arr:
        #             arr.append(label_tensor[i][j])
        # print("label_tensor arr : " + str(arr))

        # arr = []
        # row, col = final_label_tensor.shape
        # for i in range(row):
        #     for j in range(col):
        #         if final_label_tensor[i][j] not in arr:
        #             arr.append(final_label_tensor[i][j])
        # print("final_label_tensor arr : " + str(arr))

        # arr = []
        # row, col = label.shape
        # for i in range(row):
        #     for j in range(col):
        #         if label[i][j] not in arr:
        #             arr.append(label[i][j])
        # print("label arr : " + str(arr))

        # cv2.imshow("label_tensor", label_tensor)
        # cv2.imshow("label", label)
        # cv2.waitKey(0)

        # for j in range(len(self.cld)):
        #     # print("label" + str(j))
        #     print(label_tensor[j])

        # cv2.imshow("label", label)
        # cv2.waitKey(0)
        
        return torch.from_numpy(img_masked.astype(np.float32)), \
               torch.from_numpy(label_tensor.astype(int))
               # torch.from_numpy(final_label_tensor.astype(int))
               # torch.from_numpy(label.astype(int))

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
