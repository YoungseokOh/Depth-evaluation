# Copyright Nextchip 2021. All rights reserved.
import os, random
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import mode


def write_coef(path, coef, degree):
    save_path = path + os.path.join('/degree_{}_coef'.format(degree))
    if not os.path.isfile(save_path):
        np.save(save_path, coef)


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def read_folder_list(path):
    folder_list = os.listdir(path)
    return folder_list


def check_exist(path):
    return os.path.exists(path)


def make_folder(path):
    return os.makedirs(path)


def check_folder(path):
    if not check_exist(path):
        make_folder(path)
    return True


def rand_img(path):
    return random.choice([file for file in os.listdir(path) if file.endswith('.png') or file.endswith('.jpg')])


def img_read(path):
    return cv2.imread(path, 0)


def img_resize(img, width, height):
    return cv2.resize(img, [width, height])


def cal_distance(depth, coef):
    coef_len = len(coef)
    feat_depth = []
    for i in range(coef_len):
        feat_depth.append(pow(depth, i))
    return round(sum(feat_depth * coef), 2)


def poly_feature(df_feature, degree=2):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(df_feature)
    return X_poly, poly_reg


def melt_to_col(depth_name_list, df_depth_reggresion):
    reg_depth_error = []
    reg_depth_error_rate = []
    reg_depth_error_rate_avg = []
    reg_depth = []
    index_list = df_depth_reggresion.index
    for variable in depth_name_list:
        reg_depth_error_temp = []
        reg_depth_error_rate_temp = []
        reg_depth_diff_temp = []
        for index in index_list:
            if str(variable) == str(df_depth_reggresion['variable'][index]):
                reg_depth_error_temp.append(
                    abs(float(df_depth_reggresion['regression'][index]) - float(
                        df_depth_reggresion['variable'][index])))
                reg_depth_error_rate_temp.append((
                                                         abs(float(df_depth_reggresion['regression'][index]) - float(
                                                             df_depth_reggresion['variable'][index])) / float(
                                                     df_depth_reggresion['variable'][index])) * 100)
                reg_depth_diff_temp.append(abs(float(df_depth_reggresion['regression'][index]) - float(variable)))
        reg_depth_error.append(reg_depth_error_temp)
        reg_depth_error_rate.append(reg_depth_error_rate_temp)
        reg_depth.append(reg_depth_diff_temp)
        reg_depth_error_rate_avg.append(sum(reg_depth_error_rate_temp) / len(reg_depth_error_rate_temp))
    return reg_depth_error, reg_depth_error_rate, reg_depth_error_rate_avg, reg_depth


def create_dataframe(depth, name_list):
    return pd.DataFrame(depth, name_list)


def paste_path(main_path, paste_path):
    return os.path.join('/', main_path, paste_path)


def search_txt_file(path, category):
    file_list = read_folder_list(path)
    for file in file_list:
        filename, ext = os.path.splitext(file)
        if ext == '.txt' and filename[-len(category):] == category:
            return file


def search_npy_file(path):
    file_list = read_folder_list(path)
    for file in file_list:
        filename, ext = os.path.splitext(file)
        if ext == '.npy':
            return file


def df_to_avg_list(depth_list):
    avg_list = []
    for depth in range(len(depth_list)):
        temp_depth_list = list(depth_list.iloc[depth].values)
        temp_depth_list = [x for x in temp_depth_list if x == x]
        avg_list.append(sum(temp_depth_list) / len(temp_depth_list))
    return avg_list


def df_to_error_rate_list(depth_list):
    avg_list = []
    for depth in range(len(depth_list)):
        temp_depth_list = list(depth_list.iloc[depth].values)
        temp_depth_list = [(x / int(depth_list.index[depth])) for x in temp_depth_list if x == x]
        avg_list.append((sum(temp_depth_list) / len(temp_depth_list)) * 100)
    return avg_list


def df_option_to_list(depth_list, option='mode'):
    mode_list = []
    for depth in range(len(depth_list)):
        temp_depth_list = list(depth_list.iloc[depth].values)
        temp_depth_list = [x for x in temp_depth_list if x == x]
        if option == 'mode':
            temp_depth_list = mode(temp_depth_list)
            mode_list.append(temp_depth_list[0][0])
        elif option == 'avg':
            temp_depth_list = np.average(temp_depth_list)
            mode_list.append(round(temp_depth_list, 0))
        elif option == 'median':
            temp_depth_list = np.median(temp_depth_list)
            mode_list.append(round(temp_depth_list, 0))
    return mode_list


def box_coord_to_list(box_str_list):
    global box_string
    if box_str_list:
        box_string = box_str_list[len(box_str_list)-1]
    coord_temp_list = []
    temp_c = ''
    for c in box_string:
        if c != ' ':
            temp_c = temp_c + c
        else:
            coord_temp_list.append(int(temp_c))
            temp_c = ''
    return coord_temp_list


def split_str_in_list(depth_box_coord):
    coord_list = []
    for i in range(0, len(depth_box_coord)):
        print(depth_box_coord[i])
        coord_list.append(tuple(depth_box_coord[i].split(' ')[:-1]))
    return coord_list


def coord_to_depth(path, coord):
    depth = cv2.imread(path, cv2.CV_16U)
    depth_list = []
    for i in coord:
        depth_list_temp = []
        for num in range(0, len(i)):
            depth_list_temp.append(depth[int(i[num][1])][int(i[num][0])])
        depth_list.append(depth_list_temp)
    return depth_list


def load_depth_list(path, gt_dist, folder_name_list, scale_num=30, ground=None):
    depth_list = []
    bottom_depth_list = []
    depth_diff = []
    depth_name_list = []
    box_coord = []
    if ground:
        for folder in folder_name_list:
            exp_path = path + os.path.join('/', folder)
            print(folder)
            # txt
            depthtxt_name = exp_path + os.path.join('/', folder) + '_depth.txt'
            depth_box_coord = exp_path + os.path.join('/', folder) + '.txt'
            # read_csv
            depth_temp = pd.read_csv(depthtxt_name, delimiter=',').T
            depth_box_coord_temp = pd.read_csv(depth_box_coord, delimiter=',', header=None)
            # list map
            depth_temp = list(map(float, depth_temp.index.values))
            depth_box_coord_temp = list(map(str, depth_box_coord_temp[0]))
            depth_box_coord_temp = split_str_in_list(depth_box_coord_temp)
            # append
            depth_list.append(depth_temp)
            box_coord.append(depth_box_coord_temp)
        depth_name_list = list(map(int, folder_name_list))
        # df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
        return depth_list, bottom_depth_list, depth_name_list, depth_diff, box_coord
    elif not ground:
        for folder in folder_name_list:
            exp_path = path + os.path.join('/', folder)
            if check_folder(exp_path):
                meter_list = gt_dist[folder].split(" ")
            for meter in meter_list:
                # txt
                depthtxt_name = exp_path + os.path.join('/', meter) + '_depth.txt'
                bottom_depthtxt_name = exp_path + os.path.join('/', meter) + '_bottom_depth.txt'
                depth_box_coord = exp_path + os.path.join('/', meter) + '.txt'
                # read_csv
                depth_temp = pd.read_csv(depthtxt_name, delimiter=',').T
                bottom_depth_temp = pd.read_csv(bottom_depthtxt_name, delimiter=',').T
                depth_box_coord_temp = pd.read_csv(depth_box_coord, delimiter=',').T
                # list map
                depth_temp = list(map(float, depth_temp.index.values))
                bottom_depth_temp = list(map(float, bottom_depth_temp.index.values))
                depth_box_coord_temp = list(map(str, depth_box_coord_temp.index.values))
                depth_box_coord_temp = box_coord_to_list(depth_box_coord_temp)
                coord_temp = []
                for y in range(depth_box_coord_temp[3], depth_box_coord_temp[1]):
                    for x in range(depth_box_coord_temp[2], depth_box_coord_temp[0]):
                        coord_temp.append((y, x))
                # append
                depth_list.append(depth_temp)
                box_coord.append(coord_temp)
                bottom_depth_list.append(bottom_depth_temp)
                depth_diff_temp = []
                for depth in depth_temp:
                    diff_temp = abs((int(depth) / scale_num) - float(meter))
                    depth_diff_temp.append(diff_temp)
                depth_diff.append(depth_diff_temp)
                depth_name_list.append(meter)
        depth_name_list = list(map(int, depth_name_list))
        # df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
        return depth_list, bottom_depth_list, depth_name_list, depth_diff, box_coord


def save_depth_list(data, meter_list, exp_path, depth_data, ground=None):
    return 0
    # if ground:
    #     savetxt_name = exp_path + os.path.join('/', meter_list[0]) + '.txt'
    #     depthtxt_name = exp_path + os.path.join('/', meter_list[0]) + '_depth.txt'
    #     bottom_depthtxt_name = exp_path + os.path.join('/', meter_list[0]) + '_bottom_depth.txt'
    #     if not os.path.isfile(savetxt_name) or os.path.isfile(depthtxt_name):
    #         p_f = open(savetxt_name, 'w')
    #         for n in data.file_manager.box_data[count]:
    #             p_f.write(''.join(str(n)) + ' ')
    #         p_f.close()
    #         depth_list = data.file_manager.depth_data