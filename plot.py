import pandas as pd
from matplotlib import pyplot as plt
import config
import os
import utils
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor

data = config.config()

def cal_error_regression(depth_name_list, df_regression):
    reg_depth_error, reg_depth_error_rate, reg_depth_error_rate_avg, reg_depth = utils.melt_to_col(depth_name_list,
                                                                                                   df_regression)
    reg_depth_diff = utils.create_dataframe(reg_depth_error, depth_name_list)
    reg_depth_reg = utils.create_dataframe(reg_depth, depth_name_list)
    df_depth_reg_error_rate = utils.df_to_error_rate_list(reg_depth_reg)
    return reg_depth_diff, df_depth_reg_error_rate


def set_regression(depth_list, depth_name_list, degree=2):
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)
    df_feature, poly_leg = utils.poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), degree)
    lin_reg_2 = LinearRegression()
    # lin_reg_2 = Ridge()
    lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    utils.write_coef(data.file_manager.save_path, np.array(lin_reg_2.coef_), degree)
    # lin_reg_2 = RANSACRegressor(random_state=0).fit(df_feature, df_depth_reggresion['variable'].values)
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
    plt.grid(True)
    plt.title('average error rate : {}%'.format(round(float(sum(df_rate) / len(df_rate)), 2)))
    plt.show()


def main():
    # select category
    #######
    category = input("Select category [box, car, road, person, ground, test] :")
    if category == 'ground':
        direction_ewns = input("Select direction [east, west, north, south] :")
    #######
    if category == 'ground':
        data.file_manager.path_category_update(category, direction_ewns)
        if not utils.check_exist(data.file_manager.regression_data + os.path.join('/', category)):
            utils.make_folder(data.file_manager.regression_data + os.path.join('/', category))
    else:
        data.file_manager.path_category_update(category)
        if not utils.check_exist(data.file_manager.regression_data + os.path.join('/', category)):
            utils.make_folder(data.file_manager.regression_data + os.path.join('/', category))
    #######
    gt_dist = data.distance.read_gt_distance(utils.read_folder_list(data.file_manager.img_path),
                                             data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    scale = data.file_manager.scale_num
    degree = data.file_manager.degree_num
    depth_list, bottom_depth_list, depth_name_list, depth_diff, box_coord = utils.load_depth_list(
        data.file_manager.save_path,
        gt_dist, folder_names,
        scale,
        ground=True)
    folder_names_list = list(folder_names)
    selected_dist = folder_names_list[-1]
    data.file_manager.depth_file = utils.rand_img(data.file_manager.depth_path + os.path.join('/', selected_dist))
    selected_dist_path = data.file_manager.depth_path + os.path.join('/', selected_dist, data.file_manager.depth_file)
    selected_depth_list = utils.coord_to_depth(selected_dist_path, box_coord)

    # -_-# original #-_-#
    df_depth_regression = set_regression(depth_list, depth_name_list, degree=degree)
    df_reg_diff_ori, df_error_rate_ori = cal_error_regression(depth_name_list, df_depth_regression)
    plt.scatter(df_depth_regression['variable'].values, df_depth_regression['value'].values, c="orange", alpha=0.5)
    boxplot(df_reg_diff_ori, df_error_rate_ori)
    df_depth_regression.to_csv(
        data.file_manager.regression_data + os.path.join('/', category) +
        os.path.join('/', 'regression_original_{}.csv'.format(degree)))
    plt.show()

    # -_-# mode #-_-#
    df_depth_mode_list = utils.df_option_to_list(utils.create_dataframe(depth_list, depth_name_list))
    df_depth_mode_regression = set_regression(df_depth_mode_list, depth_name_list)
    df_reg_diff_mode, df_error_rate_mode = cal_error_regression(depth_name_list, df_depth_mode_regression)
    plt.scatter(df_depth_mode_regression['variable'].values, df_depth_mode_regression['value'].values)
    df_depth_mode_regression.to_csv(
        data.file_manager.regression_data + os.path.join('/', category) +
        os.path.join('/', 'regression_mode_{}.csv'.format(degree)))
    boxplot(df_reg_diff_mode, df_error_rate_mode)

    # -_-# bottom_line #-_-#
    # df_depth_bottom_list = utils.df_option_to_list(utils.create_dataframe(bottom_depth_list, depth_name_list))
    # df_depth_bottom_regression = set_regression(bottom_depth_list, depth_name_list)
    # df_reg_diff_mode, df_error_rate_mode = cal_error_regression(depth_name_list, df_depth_bottom_regression)
    # plt.scatter(df_depth_bottom_regression['variable'].values, df_depth_bottom_regression['value'].values)
    # df_depth_bottom_regression.to_csv(
    #     data.file_manager.regression_data + os.path.join('/', category) +
    #     os.path.join('/', 'regression_bottom_line_{}.csv'.format(degree)))
    # boxplot(df_reg_diff_mode, df_error_rate_mode)

    # -_-# option plot #-_-#
    df_depth_mode_list = utils.df_option_to_list(utils.create_dataframe(bottom_depth_list, depth_name_list))
    df_depth_avg_list = utils.df_option_to_list(utils.create_dataframe(bottom_depth_list, depth_name_list), 'avg')
    df_depth_median_list = utils.df_option_to_list(utils.create_dataframe(bottom_depth_list, depth_name_list), 'median')
    df_depth_mode_regression = set_regression(df_depth_mode_list, depth_name_list, degree=degree)
    df_depth_avg_regression = set_regression(df_depth_avg_list, depth_name_list, degree=degree)
    df_depth_median_regression = set_regression(df_depth_median_list, depth_name_list, degree=degree)
    plt.scatter(df_depth_bottom_regression['variable'].values, df_depth_bottom_regression['value'].values, c="orange",
                alpha=0.5)
    plt.plot(df_depth_mode_regression['variable'].values, df_depth_mode_regression['value'].values, 'o',
             linestyle='dashed', linewidth=2, markersize=6, alpha=.8)
    plt.plot(df_depth_avg_regression['variable'].values, df_depth_avg_regression['value'].values, 'o',
             linestyle='dashed', linewidth=2, markersize=6, alpha=.8, c="red")
    plt.plot(df_depth_median_regression['variable'].values, df_depth_median_regression['value'].values, 'o',
             linestyle='dashed', linewidth=2, markersize=6, alpha=.8, c="green")
    plt.legend(['mode', 'avg', 'median'])
    plt.grid(True)
    plt.xticks(depth_name_list)
    plt.xlabel('depth(m)')
    plt.ylabel('value')
    plt.show()
    df_reg_diff_mode, df_error_rate_mode = cal_error_regression(depth_name_list, df_depth_mode_regression)
    df_reg_diff_avg, df_error_rate_avg = cal_error_regression(depth_name_list, df_depth_avg_regression)
    df_reg_diff_median, df_error_rate_median = cal_error_regression(depth_name_list, df_depth_median_regression)
    boxplot(df_reg_diff_mode, df_error_rate_mode)
    boxplot(df_reg_diff_avg, df_error_rate_avg)
    boxplot(df_reg_diff_median, df_error_rate_median)

    # -_-# scaled #-_-#
    # df_scaled_depth_reggresion = pd.DataFrame(depth_diff, depth_name_list).T.melt().dropna(axis=0)
    # plt.scatter(df_scaled_depth_reggresion['variable'].values, df_scaled_depth_reggresion['value'].values)
    # df_scaled_depth_reggresion['regression'] = df_scaled_depth_reggresion['value']
    # df_reg_diff_scaled, df_error_rate_scaled = cal_error_regression(depth_name_list, df_scaled_depth_reggresion)
    # boxplot(df_reg_diff_scaled, df_error_rate_scaled)

    # -_-# degree of polynomial test #-_-#
    # poly_test = []
    # for i in range(1, 10):
    #     df_depth_regression = set_regression(bottom_depth_list, depth_name_list, i)
    #     df_reg_diff_ori, df_error_rate_ori = cal_error_regression(depth_name_list, df_depth_regression)
    #     poly_test.append(round(float(sum(df_error_rate_ori) / len(df_error_rate_ori)), 2))
    # plt.plot(poly_test)
    # plt.title('Evaluation for degree of polynomial')
    # plt.xlabel('degree')
    # plt.ylabel('avg. error(%)')
    # plt.grid(True)
    # plt.show()
    print('done')


if __name__ == "__main__":
    main()
