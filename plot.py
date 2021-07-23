import pandas as pd
from matplotlib import pyplot as plt
import config
import os
import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = config.config()


def boxplot(df_diff):
    df_mean = df_diff.T.mean()
    ax = df_diff.T.plot(kind='box')
    ax.plot(ax.get_xticks(), df_mean, color='blue', linewidth=0.5, marker='o', markersize=0.8)
    plt.xlabel('depth(m)')
    plt.ylabel('error(m)')
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
    scale_num = 35
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
                diff_temp = abs((int(depth) / 35) - float(meter))
                depth_diff_temp.append(diff_temp)
            depth_diff.append(depth_diff_temp)
            depth_name_list.append(meter)
    df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)
    df_feature, poly_leg = utils.poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), 3)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    # plt.scatter(df_depth_reggresion['value'].values, df_depth_reggresion['variable'].values)
    plt.plot(df_depth_reggresion['value'].values,
             lin_reg_2.predict(poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1))),
             color='blue')
    df_depth_reggresion['value'] = lin_reg_2.predict(
        poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    reg_depth_error = utils.melt_to_col(depth_name_list, df_depth_reggresion)
    reg_depth_diff = utils.create_Dataframe(reg_depth_error, depth_name_list)

    # bottom depth reggression
    bottom_df_depth_reggresion = pd.DataFrame(bottom_depth_list, depth_name_list).T.melt().dropna(axis=0)
    bottom_df_feature, bottom_poly_leg = utils.poly_feature(bottom_df_depth_reggresion['value'].values.reshape(-1, 1), 3)
    lin_reg_bottom = LinearRegression()
    lin_reg_bottom.fit(bottom_df_feature, bottom_df_depth_reggresion['variable'].values)
    # plt.scatter(df_depth_reggresion['value'].values, df_depth_reggresion['variable'].values)
    plt.plot(bottom_df_depth_reggresion['value'].values,
             lin_reg_bottom.predict(bottom_poly_leg.fit_transform(bottom_df_depth_reggresion['value'].values.reshape(-1, 1))),
             color='blue')
    bottom_df_depth_reggresion['value'] = lin_reg_bottom.predict(
        bottom_poly_leg.fit_transform(bottom_df_depth_reggresion['value'].values.reshape(-1, 1)))
    bottom_reg_depth_error = utils.melt_to_col(depth_name_list, bottom_df_depth_reggresion)
    bottom_reg_depth_diff = utils.create_Dataframe(bottom_reg_depth_error, depth_name_list)
    # Box plot
    # boxplot(df_depth_diff)
    boxplot(reg_depth_diff)
    boxplot(bottom_reg_depth_diff)

    print('done')


if __name__ == "__main__":
    main()
