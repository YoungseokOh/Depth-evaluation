# Copyright Nextchip 2021. All rights reserved.
import os, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import mode


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
    return 0


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


def df_to_mode_list(depth_list):
    mode_list = []
    for depth in range(len(depth_list)):
        temp_depth_list = list(depth_list.iloc[depth].values)
        temp_depth_list = [x for x in temp_depth_list if x == x]
        temp_depth_list = mode(temp_depth_list)
        mode_list.append(temp_depth_list[0][0])
    return mode_list


def load_depth_list(path, gt_dist, folder_name_list, scale_num=30):
    depth_list = []
    bottom_depth_list = []
    depth_diff = []
    depth_name_list = []
    for folder in folder_name_list:
        exp_path = path + os.path.join('/', folder)
        if check_folder(exp_path):
            meter_list = gt_dist[folder].split(" ")
        depth_temp = []
        bottom_depth_temp = []
        depth_diff_temp = []
        for meter in meter_list:
            depthtxt_name = exp_path + os.path.join('/', meter) + '_depth.txt'
            bottom_depthtxt_name = exp_path + os.path.join('/', meter) + '_bottom_depth.txt'
            depth_temp = pd.read_csv(depthtxt_name, delimiter=',').T
            bottom_depth_temp = pd.read_csv(bottom_depthtxt_name, delimiter=',').T
            depth_temp = list(map(float, depth_temp.index.values))
            bottom_depth_temp = list(map(float, bottom_depth_temp.index.values))
            depth_list.append(depth_temp)
            bottom_depth_list.append(bottom_depth_temp)
            depth_diff_temp = []
            for depth in depth_temp:
                diff_temp = abs((int(depth) / scale_num) - float(meter))
                depth_diff_temp.append(diff_temp)
            depth_diff.append(depth_diff_temp)
            depth_name_list.append(meter)
    depth_name_list = list(map(int, depth_name_list))
    # df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
    return depth_list, depth_name_list, depth_diff
