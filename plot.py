import pandas as pd
from matplotlib import pyplot as plt
import config
import os
import utils
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = config.config()


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
    depth_list = []
    bottom_depth_list = []
    depth_diff = []
    depth_name_list = []
    scale_num = 38

    for folder in folder_names:
        exp_path = data.file_manager.save_path + os.path.join('/', folder)
        if utils.check_folder(exp_path):
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
    df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
    df_depth_diff_error_avg = utils.df_to_avg_list(df_depth_diff)
    df_depth_diff_error_rate = utils.df_to_error_rate_list(df_depth_diff)

    # mode feature for regression
    # df_depth_mode = pd.DataFrame(depth_list, depth_name_list)
    # df_depth_mode_list = utils.df_to_mode_list(df_depth_mode)
    # df_depth_mode_regression = pd.DataFrame(df_depth_mode_list, depth_name_list).T.melt().dropna(axis=0)
    # df_mode_feature, poly_leg_mode = utils.poly_feature(df_depth_mode_regression['value'].values.reshape(-1, 1), 4)
    # lin_reg_mode = LinearRegression()
    # lin_reg_mode.fit(df_mode_feature, df_depth_mode_regression['variable'].values)
    # plt.scatter(df_depth_mode_regression['variable'].values, df_depth_mode_regression['value'].values)
    # predict_Y = lin_reg_mode.predict(poly_leg_mode.fit_transform(df_depth_mode_regression['value'].values.reshape(-1, 1)))
    # plt.scatter(predict_Y, df_depth_mode_regression['value'].values, color='blue', s=1)
    # plt.xlabel('depth(m)')
    # plt.ylabel('value')
    # df_depth_mode_regression['regression'] = lin_reg_mode.predict(
    #     poly_leg_mode.fit_transform(df_depth_mode_regression['value'].values.reshape(-1, 1)))
    # reg_depth_error_mode, reg_depth_error_rate_mode, reg_depth_error_rate_avg_mode, reg_depth_mode = utils.melt_to_col(depth_name_list, df_depth_mode_regression)
    # reg_depth_diff_mode = utils.create_Dataframe(reg_depth_error_mode, depth_name_list)
    # reg_depth_diff_rate_mode = utils.create_Dataframe(reg_depth_error_rate_mode, depth_name_list)
    # reg_depth_diff_rate_avg_mode = utils.create_Dataframe(reg_depth_error_rate_avg_mode, depth_name_list)
    # reg_depth_reg_mode = utils.create_Dataframe(reg_depth_mode, depth_name_list)
    # df_depth_reg_error_rate_mode = utils.df_to_error_rate_list(reg_depth_reg_mode)
    # df_depth_reg_error_avg_mode = utils.df_to_avg_list(reg_depth_reg_mode)
    # boxplot(reg_depth_diff_mode, df_depth_reg_error_rate_mode)

    # original df feature
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)
    df_feature, poly_leg = utils.poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), 6)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    plt.scatter(df_depth_reggresion['variable'].values, df_depth_reggresion['value'].values)
    predict_Y = lin_reg_2.predict(poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    plt.scatter(predict_Y, df_depth_reggresion['value'].values, color='blue', s=1)
    plt.xlabel('depth(m)')
    plt.ylabel('value')
    df_depth_reggresion['regression'] = lin_reg_2.predict(poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    reg_depth_error, reg_depth_error_rate, reg_depth_error_rate_avg, reg_depth = utils.melt_to_col(depth_name_list, df_depth_reggresion)
    reg_depth_diff = utils.create_Dataframe(reg_depth_error, depth_name_list)
    reg_depth_diff_rate = utils.create_Dataframe(reg_depth_error_rate, depth_name_list)
    reg_depth_diff_rate_avg = utils.create_Dataframe(reg_depth_error_rate_avg, depth_name_list)
    reg_depth_reg = utils.create_Dataframe(reg_depth, depth_name_list)
    df_depth_reg_error_rate = utils.df_to_error_rate_list(reg_depth_reg)
    df_depth_reg_error_avg = utils.df_to_avg_list(reg_depth_reg)

    # scaled df feature
    # df_scaled_depth_reggresion = pd.DataFrame(depth_diff, depth_name_list).T.melt().dropna(axis=0)
    # plt.scatter(df_scaled_depth_reggresion['variable'].values, df_scaled_depth_reggresion['value'].values)
    # df_scaled_depth_reggresion['regression'] = df_scaled_depth_reggresion['value']
    # reg_scaled_depth_error, reg_scaled_depth_error_rate, reg_scaled_depth_error_rate_avg, reg_scaled_depth = utils.melt_to_col(depth_name_list, df_scaled_depth_reggresion)
    # reg_scaled_depth_diff = utils.create_Dataframe(reg_scaled_depth_error, depth_name_list)
    # reg_scaled_depth_diff_rate = utils.create_Dataframe(reg_scaled_depth_error_rate, depth_name_list)
    # reg_scaled_depth_diff_rate_avg = utils.create_Dataframe(reg_scaled_depth_error_rate_avg, depth_name_list)
    # reg_scaled_depth_reg = utils.create_Dataframe(reg_scaled_depth, depth_name_list)
    # df_scaled_depth_reg_error_rate = utils.df_to_error_rate_list(reg_scaled_depth_reg)
    # df_scaled_depth_reg_error_avg = utils.df_to_avg_list(reg_scaled_depth_reg)

    ### degree of polynomial test ###
    # poly_test = []
    # for i in range(1, 10):
    #     df_feature, poly_leg = utils.poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), i)
    #     ############
    #     lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    #     df_depth_reggresion['regression'] = lin_reg_2.predict(
    #         poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    #     reg_depth_error, reg_depth_error_rate, reg_depth_error_rate_avg, reg_depth = utils.melt_to_col(depth_name_list, df_depth_reggresion)
    #     reg_depth_diff = utils.create_Dataframe(reg_depth_error, depth_name_list)
    #     reg_depth_diff_rate = utils.create_Dataframe(reg_depth_error_rate, depth_name_list)
    #     reg_depth_diff_rate_avg = utils.create_Dataframe(reg_depth_error_rate_avg, depth_name_list)
    #     reg_depth_reg = utils.create_Dataframe(reg_depth, depth_name_list)
    #     df_depth_reg_error_rate = utils.df_to_error_rate_list(reg_depth_reg)
    #     df_depth_reg_error_avg = utils.df_to_avg_list(reg_depth_reg)
    #     poly_test.append(round(float(sum(df_depth_reg_error_rate) / len(df_depth_reg_error_rate)), 2))
    # plt.plot(poly_test)
    # plt.xlabel('degree of polynomial')
    # plt.ylabel('avg. error(%)')


    # bottom depth reggression
    # bottom_df_depth_reggresion = pd.DataFrame(bottom_depth_list, depth_name_list).T.melt().dropna(axis=0)
    # bottom_df_feature, bottom_poly_leg = utils.poly_feature(bottom_df_depth_reggresion['value'].values.reshape(-1, 1), 3)
    # lin_reg_bottom = LinearRegression()
    # lin_reg_bottom.fit(bottom_df_feature, bottom_df_depth_reggresion['variable'].values)
    # # plt.scatter(df_depth_reggresion['value'].values, df_depth_reggresion['variable'].values)
    # plt.plot(bottom_df_depth_reggresion['value'].values,
    #          lin_reg_bottom.predict(bottom_poly_leg.fit_transform(bottom_df_depth_reggresion['value'].values.reshape(-1, 1))),
    #          color='blue')
    # bottom_df_depth_reggresion['value'] = lin_reg_bottom.predict(
    #     bottom_poly_leg.fit_transform(bottom_df_depth_reggresion['value'].values.reshape(-1, 1)))
    # bottom_reg_depth_error, bottom_reg_depth_error_rate, bottom_reg_depth_error_rate_avg = utils.melt_to_col(depth_name_list, bottom_df_depth_reggresion)
    # bottom_reg_depth_diff = utils.create_Dataframe(bottom_reg_depth_error, depth_name_list)
    # reg_depth_diff_rate = utils.create_Dataframe(bottom_reg_depth_error_rate, depth_name_list)
    # bottom_reg_depth_diff_rate_avg = utils.create_Dataframe(bottom_reg_depth_error_rate_avg, depth_name_list)
    # Box plot
    # boxplot(df_depth_diff)
    # regression
    boxplot(reg_depth_diff, df_depth_reg_error_rate)
    # scaled
    # boxplot(reg_scaled_depth_diff, df_scaled_depth_reg_error_rate)
    # original
    # boxplot(df_depth_diff, df_depth_diff_error_rate)
    # boxplot(bottom_reg_depth_diff, bottom_reg_depth_diff_rate_avg)
    print('done')


if __name__ == "__main__":
    main()
