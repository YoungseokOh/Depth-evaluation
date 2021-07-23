# Copyright Nextchip 2021. All rights reserved.
import os, random

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


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
    return random.choice(os.listdir(path))


def img_read(path):
    return 0


def poly_feature(df_feature, degree=2):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(df_feature)
    return X_poly, poly_reg


def melt_to_col(depth_name_list, df_depth_reggresion):
    reg_depth_error = []
    index_list = df_depth_reggresion.index
    for variable in depth_name_list:
        reg_depth_error_temp = []
        for index in index_list:
            if str(variable) == str(df_depth_reggresion['variable'][index]):
                reg_depth_error_temp.append(abs(float(df_depth_reggresion['variable'][index]) - float(df_depth_reggresion['value'][index])))
        reg_depth_error.append(reg_depth_error_temp)
    return reg_depth_error


def create_Dataframe(depth, name_list):
    return pd.DataFrame(depth, name_list)


def paste_path(main_path, paste_path):
    return os.path.join('/', main_path, paste_path)


def search_txt_file(path, category):
    file_list = read_folder_list(path)
    for file in file_list:
        filename, ext = os.path.splitext(file)
        if ext == '.txt' and filename[-len(category):] == category:
            return file

