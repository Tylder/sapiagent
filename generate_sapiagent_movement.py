import os
from typing import List

import numpy

from .settings import TRAINED_MODELS_FOLDER_NAME
from GUIControl.common import Int2
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import load_model


class SapiAgentMouse:

    def __init__(self, model_name="bidirectional_dx_dy_custom_unsupervised"):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        work_dir = os.path.dirname(os.path.realpath(__file__))
        training_models_path = os.path.join(work_dir, TRAINED_MODELS_FOLDER_NAME)

        model_path = os.path.join(training_models_path, model_name)

        self.model = load_model(model_path, compile=False)

    def get_movement(self, start: Int2, destination: Int2) -> [Int2]:
        array = generate_sapiagent_movement(self.model, start, destination)
        return [Int2(pos[0], pos[1]) for pos in array]


def create_equidistant_dx_dy(start: Int2, destination: Int2) -> np.ndarray:
    """
    1. Divides the distance by 128.
    2. Creates list of dx and dy equidistant
    3. Concatenates dx and dy so that the list is 128 + 128 = 256 long
    """
    length = start.diff(destination).magnitude()

    x1 = start.x
    x2 = destination.x
    y1 = start.y
    y2 = destination.y

    n = int(min(length, 128))  # max 128
    dx = (x2 - x1) / n
    dy = (y2 - y1) / n

    # x, y coordinates - floats
    x = []
    y = []
    for i in range(0, n + 1):
        x.append(x1 + i * dx)
        y.append(y1 + i * dy)
    # x, y coordinates - integers
    x_int = [int(a) for a in x]
    y_int = [int(a) for a in y]

    # first order differential
    dx_list = np.diff(x_int).tolist()
    dy_list = np.diff(y_int).tolist()

    # 0 padding
    dx_list.extend([0] * (128 - n))
    dy_list.extend([0] * (128 - n))

    # full list contains dx_list + dy_list
    full_list: List[int] = dx_list
    full_list.extend(dy_list)

    return np.array(full_list)


def scale_prediction_dx_dy_to_movement(prediction_rel_pos_array: np.ndarray, start: Int2, destination: Int2) -> np.ndarray:
    """
    SapiAgent returns a list of dx, dy which is not scaled properly to actually reach the given destination
    We therefore need to scale the result so that we get to the destination
    """
    length_of_request = (destination - start).toNumpy()
    length_of_prediction = prediction_rel_pos_array[-1, :]

    scale = length_of_request / length_of_prediction
    scale = numpy.reshape(scale, (1, 2))

    return prediction_rel_pos_array * scale + start.toNumpy().reshape(1, 2)


def dx_dy_list_to_rel_pos_list(dx_dy_list: np.ndarray) -> np.ndarray:
    dx_list = dx_dy_list.T[0]
    dy_list = dx_dy_list.T[1]

    # reverse .diff
    x_list = dx_list.cumsum() - dx_list[0]
    y_list = dy_list.cumsum() - dy_list[0]

    return np.array([x_list, y_list], float).T


def generate_sapiagent_movement(model, start: Int2, destination: Int2) -> np.ndarray:
    dx_dy_list = create_equidistant_dx_dy(start=start, destination=destination)

    X = dx_dy_list.reshape(-1, 128, 2)  # include batch size, since that is how it was trained
    # generate actions using the autoencoder
    prediction_dx_dy_list = model.predict(X)[0]  # returns np.ndarray (1, prediction) only keep one
    prediction_rel_pos_array = dx_dy_list_to_rel_pos_list(prediction_dx_dy_list)

    movement_list = scale_prediction_dx_dy_to_movement(prediction_rel_pos_array=prediction_rel_pos_array, start=start,
                                                       destination=destination)

    return movement_list


if __name__ == "__main__":
    sapiagent = SapiAgentMouse()
    movement = sapiagent.get_movement(Int2(0, 0), Int2(100, 500))

