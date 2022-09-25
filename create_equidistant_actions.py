import os
import pandas as pd
import numpy as np
from typing import List
from settings import SessionType, create_directory

LENGTH = 128

def create_actions(session, output_filepath, stat_dir, dt=False):
  f_out = open(output_filepath, 'w')

  df_start_filename = 'actions_start_stop_' + session.value + '.csv'
  df_start = pd.read_csv(os.path.join(stat_dir, df_start_filename))

  startx = df_start['startx']
  starty = df_start['starty']
  stopx = df_start['stopx']
  stopy = df_start['stopy']
  length = df_start['length']
  if dt:
    time = df_start['time']

  userid = df_start['userid']
  rows, cols = df_start.shape
  for i in range(rows):
    x1 = startx[i]
    y1 = starty[i]
    x2 = stopx[i]
    y2 = stopy[i]
    n = min(length[i], 128)
    dx = (x2 - x1) / n
    dy = (y2 - y1) / n
    # dxdydt
    if dt:
      dt = (int)(time[i] / n)

    # x, y coordinates - floats
    x = []
    y = []
    for i in range(0, n + 1):
      x.append(x1 + i * dx)
      y.append(y1 + i * dy)
    # x, y coordinates - integers
    x_int = [int(a) for a in x]
    y_int = [int(a) for a in y]

    dx_list = np.diff(x_int).tolist()
    dy_list = np.diff(y_int).tolist()

    # dxdydt
    if dt:
      dt_list: List[int] = [dt] * n

    dx_list.extend([0] * (128 - n))
    dy_list.extend([0] * (128 - n))

    # dxdydt
    if dt:
      dt_list.extend([0] * (128 - n))
    d_list: List[int] = dx_list
    d_list.extend(dy_list)
    # dxdydt
    if dt:
      d_list.extend(dt_list)
    d_list.append(userid[i])
    d_list = [str(e) for e in d_list]
    # if len(d_list) != 257:
    # dxdydt
    # if len(d_list) != 385:
    #     print("{} {}".format(i, len(d_list)))
    f_out.write(",".join(d_list) + ' \n')


def create_equidistant_actions(out_dir: str = os.path.join(os.getcwd(), "saved_data", "equidistant_actions"),
                               stat_dir: str = os.path.join(os.getcwd(), "saved_data", "statistics")):
  create_directory(out_dir)
  for session_type in SessionType:
    filename = 'equidistant_' + session_type.value + '.csv'
    output_filepath = os.path.join(out_dir, filename)
    create_actions(session=session_type, output_filepath=output_filepath, stat_dir=stat_dir, dt=False)


if __name__ == "__main__":
  create_equidistant_actions()
