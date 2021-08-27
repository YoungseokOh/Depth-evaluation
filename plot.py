import pandas as pd
from matplotlib import pyplot as plt
import config
import os
import utils
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = config.config()


def cal_error_regression(depth_name_list, df_regression):
    reg_depth_error, reg_depth_error_rate, reg_depth_error_rate_avg, reg_depth = utils.melt_to_col(depth_name_list, df_regression)
    reg_depth_diff = utils.create_dataframe(reg_depth_error, depth_name_list)
    reg_depth_reg = utils.create_dataframe(reg_depth, depth_name_list)
    df_depth_reg_error_rate = utils.df_to_error_rate_list(reg_depth_reg)
    return reg_depth_diff, df_depth_reg_error_rate


def set_regression(depth_list, depth_name_list, degree=1):
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)
    df_feature, poly_leg = utils.poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), degree)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    predict_Y = lin_reg_2.predict(poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    df_depth_reggresion['regression'] = lin_reg_2.predict(
        poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    return df_depth_reggresion


def boxplot(df_diff, df_rate):
    df_mean = df_diff.T.mean()
    ax = df_diff.T.plot(kind='box')
    ax.plot(ax.get_xticks(), df_mean, color='orange', linewidth=0.5, marker='o', markersize=0.8)
    # for i in range(len(df_rate)):
    #     ax.text(ax.get_xticks()[i]-0.15, -1.5, "{}%".format(round(df_rate[i], 2)), fontsize=7.5, color='red')
    plt.xlabel('depth(m)')
    plt.ylabel('error(m)')
    plt.title('average error rate : {}%'.format(round(float(sum(df_rate) / len(df_rate)), 2)))
    plt.show()


def main():
    # select category
    #######
    category = input("Select category [box, car, person] :")
    #######
    data.file_manager.path_category_update(category)
    gt_dist = data.distance.read_gt_distance(utils.read_folder_list(data.file_manager.img_path),
                                             data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    scale_num = 27
    depth_list, depth_name_list, depth_diff = utils.load_depth_list(data.file_manager.save_path, gt_dist, folder_names,
                                                                    scale_num)

    # original
    df_depth_regression = set_regression(depth_list, depth_name_list)
    df_reg_diff_ori, df_error_rate_ori = cal_error_regression(depth_name_list, df_depth_regression)
    plt.scatter(df_depth_regression['variable'].values, df_depth_regression['value'].values)
    boxplot(df_reg_diff_ori, df_error_rate_ori)

    # mode
    df_depth_mode_list = utils.df_to_mode_list(utils.create_dataframe(depth_list, depth_name_list))
    df_depth_mode_regression = set_regression(df_depth_mode_list, depth_name_list)
    df_reg_diff_mode, df_error_rate_mode = cal_error_regression(depth_name_list, df_depth_mode_regression)
    plt.scatter(df_depth_mode_regression['variable'].values, df_depth_mode_regression['value'].values)
    boxplot(df_reg_diff_mode, df_error_rate_mode)

    # scaled
    df_scaled_depth_reggresion = pd.DataFrame(depth_diff, depth_name_list).T.melt().dropna(axis=0)
    plt.scatter(df_scaled_depth_reggresion['variable'].values, df_scaled_depth_reggresion['value'].values)
    df_scaled_depth_reggresion['regression'] = df_scaled_depth_reggresion['value']
    df_reg_diff_scaled, df_error_rate_scaled = cal_error_regression(depth_name_list, df_scaled_depth_reggresion)
    boxplot(df_reg_diff_scaled, df_error_rate_scaled)

    ### degree of polynomial test ###
    poly_test = []
    for i in range(1, 10):
        df_depth_regression = set_regression(depth_list, depth_name_list, i)
        df_reg_diff_ori, df_error_rate_ori = cal_error_regression(depth_name_list, df_depth_regression)
        poly_test.append(round(float(sum(df_error_rate_ori) / len(df_error_rate_ori)), 2))
    plt.plot(poly_test)
    plt.xlabel('degree of polynomial')
    plt.ylabel('avg. error(%)')
    plt.show()
    print('done')


if __name__ == "__main__":

    main()
