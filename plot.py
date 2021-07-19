import pandas as pd
from matplotlib import pyplot as plt
import config
import os
import utils
from sklearn.preprocessing import PolynomialFeatures

data = config.config()

def boxplot(df_diff):
    df_mean = df_diff.T.mean()
    ax = df_diff.T.plot(kind='box')
    ax.plot(ax.get_xticks(), df_mean, color='blue', linewidth=0.5, marker='o', markersize=0.8)
    plt.xlabel('depth(m)')
    plt.ylabel('error(m)')
    plt.show()

def main():
    gt_dist = data.distance.read_gt_distance(data.file_manager.exp_counts, data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    depth_list = []
    depth_diff = []
    depth_name_list = []
    scale_num = 35
    for folder in folder_names:
        exp_path = data.file_manager.save_path + os.path.join('/', folder)
        if utils.check_folder(exp_path):
            meter_list = gt_dist[folder].split(" ")
        depth_temp = []
        depth_diff_temp = []
        for meter in meter_list:
            depthtxt_name = exp_path + os.path.join('/', meter) + '_depth.txt'
            depth_temp = pd.read_csv(depthtxt_name, delimiter=',').T
            depth_temp = list(map(float, depth_temp.index.values))
            depth_list.append(depth_temp)
            depth_diff_temp = []
            for depth in depth_temp:
                diff_temp = (int(depth) / 35) - float(meter)
                depth_diff_temp.append(diff_temp)
            depth_diff.append(depth_diff_temp)
            depth_name_list.append(meter)
    df_depth_diff = pd.DataFrame(depth_diff, depth_name_list)
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)

    boxplot(df_depth_diff)
    print('done')



if __name__ == "__main__":
    main()
