import cv2
import config
import os
import utils
import numpy as np
from flags import ActionFlag, MouseFlag
import pickle
import csv
import pandas as pd

data = config.config()

def mouse_control(event, x, y, flags, data):
    data.user_interface.mouse_data.cur_x = x
    data.user_interface.mouse_data.cur_y = y
    data.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
    if event == cv2.EVENT_LBUTTONDOWN:
        data.user_interface.mouse_data.click = True
        data.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_LBUTTONDOWN

    if event == cv2.EVENT_MOUSEMOVE:
        data.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_MOVE

    if event == cv2.EVENT_LBUTTONUP:
        data.user_interface.mouse_data.click = False
        data.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_LBUTTONUP


def main():
    global data
    # select category
    #######
    category = input("Select category [box, car, road, person] :")
    #######
    data.file_manager.path_category_update(category)
    gt_dist = data.distance.read_gt_distance(utils.read_folder_list(data.file_manager.img_path),
                                             data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    # folder check
    cv2.namedWindow('image')
    cv2.namedWindow('depth')
    cv2.setMouseCallback("image", mouse_control, data)
    depth_count = 0
    save_path = data.file_manager.save_path
    for folder in folder_names:
        exp_path = save_path + os.path.join('/', folder)
        if utils.check_folder(exp_path):
            meter_list = gt_dist[folder].split(" ")
        # Image
        data.file_manager.img_file = utils.rand_img(data.file_manager.img_path + os.path.join('/', folder))
        img = cv2.imread(data.file_manager.img_path + os.path.join('/', folder, data.file_manager.img_file))
        # Depth
        data.file_manager.depth_file = utils.rand_img(data.file_manager.depth_path + os.path.join('/', folder))
        depth = cv2.imread(data.file_manager.depth_path + os.path.join('/', folder, data.file_manager.depth_file), cv2.CV_16U)
        img = cv2.resize(img, (640, 192))
        # data.draw_image = img.copy()
        # cv2.namedWindow('image')
        while(True):
            data.draw.draw_processing(data)
            data.display.display_processing(data)
            cv2.imshow('image', data.draw_image)
            cv2.imshow('depth', data.draw_depth)
            data.draw_image = img.copy()
            data.draw_depth = depth.copy()
            data.ori_depth = depth.copy()
            k = cv2.waitKey(1)
            c = cv2.waitKeyEx(5)
            c = data.config_processing(c)
            if k == 27 or data.file_manager.box_idx == len(meter_list):
                data.file_manager.box_idx = 0  # esc를 누르면 종료
                break
        count = 0
        for meter in meter_list:
            savetxt_name = exp_path + os.path.join('/', meter) + '.txt'
            depthtxt_name = exp_path + os.path.join('/', meter) + '_depth.txt'
            bottom_depthtxt_name = exp_path + os.path.join('/', meter) + '_bottom_depth.txt'
            if not os.path.isfile(savetxt_name) or os.path.isfile(depthtxt_name):
                p_f = open(savetxt_name, 'w')
                for n in data.file_manager.box_data[count]:
                    p_f.write(''.join(str(n)) + ' ')
                p_f.close()
                depth_list = data.file_manager.depth_data[count]
                if not data.file_manager.bottom_line_data:
                    pass
                else:
                    bottom_depth_list = data.file_manager.bottom_line_data[count]
                    df_bottom = pd.DataFrame(bottom_depth_list, columns=["depth"]).T
                    df_bottom.to_csv(bottom_depthtxt_name, index=None, header=None)
                # box depth save
                df = pd.DataFrame(depth_list, columns=["depth"]).T
                # bottom line save
                df.to_csv(depthtxt_name, index=None, header=None)

                count += 1
            else:
                break
        data.file_manager.depth_data = []
        data.file_manager.bottom_line_data = []
        # print('done')


if __name__ == "__main__":
    main()

    print('done')
