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
    data.mouse_event.cur_x = x
    data.mouse_event.cur_y = y
    data.mouse_event.mouse_flag = MouseFlag.MOUSE_NOTHING
    if event == cv2.EVENT_LBUTTONDOWN:
        data.mouse_event.click = True
        data.mouse_event.mouse_flag = MouseFlag.MOUSE_LBUTTONDOWN

    if event == cv2.EVENT_MOUSEMOVE:
        data.mouse_event.mouse_flag = MouseFlag.MOUSE_MOVE

    if event == cv2.EVENT_LBUTTONUP:
        data.mouse_event.click = False
        data.mouse_event.mouse_flag = MouseFlag.MOUSE_LBUTTONUP


def main():
    global data
    # select category
    #######
    category = 'road'
    #######
    data.file_manager.path_category_update(category)
    gt_dist = data.distance.read_gt_distance(utils.read_folder_list(data.file_manager.img_path),
                                             data.file_manager.gt_distance_path)
    folder_names = gt_dist.keys()
    # folder check
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", mouse_control, data)
    coef_file = utils.search_npy_file(data.file_manager.save_path)
    data.file_manager.reg_coef = np.load(data.file_manager.save_path + '/' + coef_file)
    for folder in folder_names:
        # Image
        data.file_manager.img_file = utils.rand_img(data.file_manager.img_path + os.path.join('/', folder))
        img = cv2.imread(data.file_manager.img_path + os.path.join('/', folder, data.file_manager.img_file))
        # Depth
        data.file_manager.depth_file = utils.rand_img(data.file_manager.depth_path + os.path.join('/', folder))
        depth = cv2.imread(data.file_manager.depth_path + os.path.join('/', folder, data.file_manager.depth_file),
                           cv2.CV_16U)
        img = cv2.resize(img, (640, 192))
        while(True):
            data.display.display_processing(data)
            data.read.read_processing(data)
            cv2.imshow('image', data.draw_image)
            data.draw_image = img.copy()
            data.read_depth = depth.copy()
            k = cv2.waitKey(1)
            if k == 27:  # esc를 누르면 종료
                data.draw_image = img.copy()
                break
        count = 0
        # print('done')


if __name__ == "__main__":
    main()

    print('done')
