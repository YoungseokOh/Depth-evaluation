import pandas as pd
import matplotlib
import config
import os
import utils

data = config.config()


def main():
    gt_dist = data.distance.read_gt_distance(data.file_manager.exp_counts, data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    depth_list = []
    depth_name_list = []
    for folder in folder_names:
        exp_path = data.file_manager.save_path + os.path.join('/', folder)
        if utils.check_folder(exp_path):
            meter_list = gt_dist[folder].split(" ")
        depth_temp = []
        for meter in meter_list:
            depthtxt_name = exp_path + os.path.join('/', meter) + '_depth.txt'
            depth_temp = pd.read_csv(depthtxt_name, delimiter=',').T
            depth_temp = list(map(float, depth_temp.index.values))
            depth_list.append(depth_temp)
            depth_name_list.append(meter)
    print('done')

if __name__ == "__main__":
    main()
