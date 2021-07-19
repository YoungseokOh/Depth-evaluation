from enum import Enum
import numpy as np


class ActionFlag(Enum):

        ACTION_NOTHING = 1,
        ACTION_DRAW = 2,
        ACTION_SAVE_DATA = 3,


class MouseFlag(Enum):

        MOUSE_NOTHING = 11,
        MOUSE_LBUTTONDOWN = 12,
        MOUSE_MOVE = 13,
        MOUSE_LBUTTONUP = 14,