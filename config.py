import os
from flags import ActionFlag, MouseFlag
import flags
import utils
from enum import Enum
import cv2
import numpy as np


class distance:
    def read_gt_distance(self, exp_counts, gt_dist_path):
        lines = utils.readlines(gt_dist_path)
        zip_distance = zip(exp_counts, lines)
        return dict(zip_distance)


class config:
    def __init__(self):
        self.file_manager = file_manager()
        self.distance = distance()
        self.user_interface = UserInterface()
        # self.mouse_event = mouse_event()
        self.draw = draw()
        self.read = read()
        self.display = display()
        self.draw_image = np.zeros((self.display.image_size[1].astype(np.uint32), self.display.image_size[0].astype(np.uint32), 3),
            dtype=np.uint8)
        self.draw_depth = np.zeros((self.display.image_size[1].astype(np.uint32), self.display.image_size[0].astype(np.uint32)),
            dtype=np.float32)
        self.read_depth = np.zeros((self.display.image_size[1].astype(np.uint32), self.display.image_size[0].astype(np.uint32)),
            dtype=np.float32)
        self.ori_depth = np.zeros((self.display.image_size[1].astype(np.uint32), self.display.image_size[0].astype(np.uint32)),
            dtype=np.float32)
        self.action_flag = ActionFlag.ACTION_NOTHING

    def config_processing(self, c):
        self.user_interface.keyboard_data.keyboard_event_processing(self, c)


class UserInterface(config):
    def __init__(self):
        self.mouse_data = mouse_event()
        self.keyboard_data = keyboard_event()
        self.action_flag = 0


class mouse_event(config):
    def __init__(self):
        self.x1 = self.y1 = None
        self.cur_x = self.cur_y = None
        self.click = False
        # Rectangle coord
        self.rect_x = self.rect_y = self.rect_x1 = self.rect_y1 = None
        self.mouse_flag = MouseFlag.MOUSE_NOTHING


class keyboard_event(config):
    def __init__(self):
        self.action_flag = 0


    def keyboard_event_processing(self, config, c):
        if c == 49: # '1'
            config.action_flag = ActionFlag.ACTION_BOX_DRAW
            print('Action Box_Draw!')
        if c == 50: # '2'
            config.action_flag = ActionFlag.ACTION_SPOT_DRAW
            print('Action Spot_Draw!')


class file_manager(config):
    def __init__(self):
        # paths
        img_file = ""
        depth_file = ""
        self.gt_distance_path = ""
        self.img_folder = '/img'
        self.depth_folder = '/depth'
        self.seg_folder = '/seg'
        self.save_folder = '/pixel_coord'
        self.regression_data = './regression_data'
        self.ori_path = 'Y:/monodepth_results/front_results/_evaluation/real_meter'
        self.img_path = self.ori_path + self.img_folder
        self.depth_path = self.ori_path + self.depth_folder
        self.save_path = self.ori_path + self.save_folder
        self.seg_path = self.ori_path + self.seg_folder
        # settings
        self.meter_count = 0
        self.depth_len = 0
        self.box_idx = 0
        self.reg_coef = 0
        self.scale_num = 80
        self.degree_num = 1
        self.box_data = np.zeros((9,4), dtype = np.int32)# frame 넘길 때마다 박스데이터 초기화
        self.depth_data = [] # frame 넘길 때마다 박스데이터 초기화
        self.bottom_line_data = []


    def path_category_update(self, category, ground=None):
        self.img_path = utils.paste_path(self.img_path, category)
        self.depth_path = utils.paste_path(self.depth_path, category)
        self.save_path = utils.paste_path(self.save_path, category)
        self.gt_distance_path = os.path.join('/', self.ori_path, utils.search_txt_file(self.ori_path, category))
        if ground:
            self.img_path = utils.paste_path(self.img_path, ground)
            self.depth_path = utils.paste_path(self.depth_path, ground)
            self.save_path = utils.paste_path(self.save_path, ground)


class read(config):
    def __init__(self):
        pass


    def read_processing(self, config):
        self.read_click = read.read_click(self, config)
        self.read_up = read.read_up(self, config)


    def read_click(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONDOWN:
        # if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONDOWN:
            # print(config.user_interface.mouse_data.click)
            config.user_interface.mouse_data.click = True
            config.user_interface.mouse_data.x1, config.user_interface.mouse_data.y1 = config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y
        elif config.user_interface.mouse_data.x1 == None and config.user_interface.mouse_data.y1 == None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            return None
        cv2.putText(config.draw_image, '{}(m)'.format(utils.cal_distance(config.read_depth[config.user_interface.mouse_data.y1][config.user_interface.mouse_data.x1],
                                                                         config.file_manager.reg_coef)),
                    (config.user_interface.mouse_data.x1 + 10, config.user_interface.mouse_data.y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.45, (255, 255, 255))
        # print("사각형의 왼쪽위 설정 : (" + str(config.user_interface.mouse_data.x1) + ", " + str(config.user_interface.mouse_data.y1) + ")")


    def read_up(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONUP and config.user_interface.mouse_data.x1 is not None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            config.user_interface.mouse_data.cur_x = config.user_interface.mouse_data.cur_y = config.user_interface.mouse_data.x1 = config.user_interface.mouse_data.y1 = None


class draw(config):
    def __init__(self):
        self.ratio = 0.1
        pass


    def draw_processing(self, config):
        if config.action_flag == config.action_flag.ACTION_BOX_DRAW:
            self.draw_box_click = draw.box_draw_click(self, config)
            self.draw_box_move = draw.box_draw_move(self, config)
            self.draw_box_up = draw.box_draw_up(self, config)
        if config.action_flag == config.action_flag.ACTION_SPOT_DRAW:
            self.draw_spot_click = draw.spot_draw_click(self, config)
            self.draw_spot_up = draw.spot_draw_up(self, config)


    def spot_draw_click(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONDOWN:
            config.user_interface.mouse_data.click = True
            config.user_interface.mouse_data.x1, config.user_interface.mouse_data.y1 = config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y
            # print("사각형의 왼쪽위 설정 : (" + str(config.user_interface.mouse_data.x1) + ", " + str(
            # config.user_interface.mouse_data.y1) + ")")
            cv2.circle(config.draw_image,
                       (config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y),
                       12,
                       (255, 0, 0),
                          1)
            cv2.line(config.draw_image, (0, config.user_interface.mouse_data.cur_y),
                     (640, config.user_interface.mouse_data.cur_y), (255, 255, 255), 1)
        elif config.user_interface.mouse_data.x1 == None and config.user_interface.mouse_data.y1 == None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            return None


    def spot_draw_up(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONUP and config.user_interface.mouse_data.x1 is not None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            # depth
            depth_temp = []
            depth_temp.append(config.ori_depth[config.user_interface.mouse_data.cur_y][config.user_interface.mouse_data.cur_x])
            config.file_manager.depth_data.append(depth_temp)
            config.file_manager.box_data[config.file_manager.box_idx][0] = config.user_interface.mouse_data.cur_x
            config.file_manager.box_data[config.file_manager.box_idx][1] = config.user_interface.mouse_data.cur_y
            config.file_manager.box_idx += 1
            print('num count : {}'.format(config.file_manager.box_idx))


    def box_draw_click(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONDOWN:
        # if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONDOWN:
            # print(config.user_interface.mouse_data.click)
            config.user_interface.mouse_data.click = True
            config.user_interface.mouse_data.x1, config.user_interface.mouse_data.y1 = config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y
        elif config.user_interface.mouse_data.x1 == None and config.user_interface.mouse_data.y1 == None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            return None
            # print("사각형의 왼쪽위 설정 : (" + str(config.user_interface.mouse_data.x1) + ", " + str(config.user_interface.mouse_data.y1) + ")")


    def box_draw_move(self, config):
        if config.user_interface.mouse_data.click == True and config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_MOVE:
                cv2.rectangle(config.draw_image, (config.user_interface.mouse_data.x1,config.user_interface.mouse_data.y1),
                              (config.user_interface.mouse_data.cur_x,config.user_interface.mouse_data.cur_y),(255,0,0),1)
                cv2.line(config.draw_image, (0, config.user_interface.mouse_data.cur_y), (640, config.user_interface.mouse_data.cur_y), (255, 255, 255), 1)
                cv2.rectangle(config.draw_depth, (config.user_interface.mouse_data.x1,config.user_interface.mouse_data.y1),
                              (config.user_interface.mouse_data.cur_x,config.user_interface.mouse_data.cur_y),(255,0,0),1)
                # print("(" + str(config.user_interface.mouse_data.x1) + ", " + str(config.user_interface.mouse_data.y1) + "), (" +
                #       str(config.user_interface.mouse_data.cur_x) + ", " + str(config.user_interface.mouse_data.cur_y) + ")")


    def box_draw_up(self, config):
        if config.user_interface.mouse_data.mouse_flag == MouseFlag.MOUSE_LBUTTONUP and config.user_interface.mouse_data.x1 is not None:
            config.user_interface.mouse_data.mouse_flag = MouseFlag.MOUSE_NOTHING
            cv2.rectangle(config.draw_image, (config.user_interface.mouse_data.x1, config.user_interface.mouse_data.y1),
                          (config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y), (100, 255, 0), 1)
            cv2.rectangle(config.draw_depth, (config.user_interface.mouse_data.x1, config.user_interface.mouse_data.y1),
                          (config.user_interface.mouse_data.cur_x, config.user_interface.mouse_data.cur_y), (255, 255, 0), 1)

            # image
            config.file_manager.box_data[config.file_manager.box_idx][0] = config.user_interface.mouse_data.cur_x
            config.file_manager.box_data[config.file_manager.box_idx][1] = config.user_interface.mouse_data.cur_y
            config.file_manager.box_data[config.file_manager.box_idx][2] = config.user_interface.mouse_data.x1
            config.file_manager.box_data[config.file_manager.box_idx][3] = config.user_interface.mouse_data.y1
            # depth
            x_diff = config.user_interface.mouse_data.cur_x - config.user_interface.mouse_data.x1
            y_diff = config.user_interface.mouse_data.cur_y - config.user_interface.mouse_data.y1
            config.file_manager.depth_len = x_diff * y_diff
            depth_temp = []
            depth_bottom_temp = []
            # rect depth extract
            for j in range(config.user_interface.mouse_data.y1, config.user_interface.mouse_data.cur_y):
                for i in range(config.user_interface.mouse_data.x1, config.user_interface.mouse_data.cur_x):
                    depth_temp.append(config.ori_depth[j][i])
            # bottom line depth extract
            for line in range(config.user_interface.mouse_data.x1, config.user_interface.mouse_data.cur_x):
                    depth_bottom_temp.append(config.ori_depth[config.user_interface.mouse_data.cur_y][line])
            config.file_manager.depth_data.append(depth_temp)
            config.file_manager.bottom_line_data.append(depth_bottom_temp)
            config.user_interface.mouse_data.cur_x = config.user_interface.mouse_data.cur_y = config.user_interface.mouse_data.x1 = config.user_interface.mouse_data.y1 = None
            config.file_manager.box_idx += 1


class display(config):
    def __init__(self, width = None, height = None, frame = 0):
        if width and height:
            self.image_size =  np.array([width, height]).astype(dtype=np.float32)
        else:
            self.image_size = np.array([640, 192]).astype(dtype=np.float32)

        if frame:
            self.frame = frame
        else:
            self.frame = 0
        self.far_step = 1
        self.circle_size = 3


    def display_processing(self, config, ground=None):
        if ground:
            self.spot_display = display.spot_display(self, config)
        else:
            self.box_display = display.box_display(self, config)


    def box_display(self, config):
        if config.file_manager.box_idx >= 0:
            for x in range(0, config.file_manager.box_idx):
                cur_x = config.file_manager.box_data[x][0].copy()
                cur_y = config.file_manager.box_data[x][1].copy()
                x1 = config.file_manager.box_data[x][2].copy()
                y1 = config.file_manager.box_data[x][3].copy()
                cv2.rectangle(config.draw_image, (x1, y1), (cur_x, cur_y), (100, 0, 0), 1)
                cv2.rectangle(config.draw_depth, (x1, y1), (cur_x, cur_y), (100, 0, 255), 1)
                cv2.line(config.draw_image, (x1, cur_y), (cur_x, cur_y), (255, 255, 255), 1)
                # cv2.line(config.draw_depth, (x1, cur_y), (cur_x, cur_y), (100, 255, 255), 1)


    def spot_display(self, config):
        cv2.line(config.draw_image, (0, config.user_interface.mouse_data.cur_y),
                 (640, config.user_interface.mouse_data.cur_y), (255, 255, 255), 1)
        if config.file_manager.box_idx >= 0:
            for x in range(0, config.file_manager.box_idx):
                cur_x = config.file_manager.box_data[x][0].copy()
                cur_y = config.file_manager.box_data[x][1].copy()
                cv2.circle(config.draw_image,
                           (cur_x, cur_y),
                           4,
                           (0, 0, 255),
                              1)
