from enum import Enum
import os

class ModelType(Enum):
  FCN = 0
  BI_DIRECTIONAL = 1


class LossFuncType(Enum):
  CUSTOM = 0
  MSE = 1


class ActionType(Enum):
  # Point Click
  PC = 0
  # Drag Drop
  DD = 1


class SessionType(Enum):
  MIN1 = '1min'
  MIN3 = '3min'


class FeatureType(Enum):
  DX_DY = 'dx_dy'
  DX_DY_DT = 'dx_dy_dt'


def create_directory(directory: str):
  try:
    os.makedirs(directory)
  except OSError as e:
    print(e)
    print('Directory %s already exists' % directory)
  else:
    print('Successfully created the directory %s' % directory)

def get_name(model_type: ModelType, loss_func_type: LossFuncType, is_supervised: bool, suffix="dx_dy") -> str:
  type_name = 'fcn' if model_type == ModelType.FCN else 'bidirectional'

  loss_func_type_name = ""
  if loss_func_type == LossFuncType.CUSTOM:
    loss_func_type_name = "custom"
  elif loss_func_type == LossFuncType.MSE:
    loss_func_type_name = "mse"

  supervised_name = 'supervised' if is_supervised else 'unsupervised'

  return type_name + "_" + suffix + "_" + loss_func_type_name + "_" + supervised_name


OUTPUT_FIGURES = "output_png"
TRAINING_CURVES_FOLDER_NAME = "TRAINING_CURVES"
TRAINED_MODELS_FOLDER_NAME = "TRAINED_MODELS"

# Init random generator
RANDOM_STATE = 11235

# Model parameters
# BATCH_SIZE = 32
# EPOCHS = 100

# Temporary filename - used to save ROC curve data
TEMP_NAME = "scores.csv"

# Input shape MOUSE - SapiMouse
# FEATURES = 128
# DIMENSIONS = 2

# Other parameters
TRAINING = True
# 'fcn', 'bidirectional'
# KEY = "fcn" # Fully Convolutional Network
# 'mse' 'custom'
# LOSS = "mse"


# 'supervised', 'supervised'
# TRAINING_TYPE = "unsupervised"
# OUTPUT_PNG = OUTPUT_FIGURES + "/" + KEY + "_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE
#
# # model names
# model_names = {
#     "fcn": "fcn_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE + ".h5",
#     "bidirectional": "bidirectional_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE + ".h5",
# }

# number of plots
NUM_PLOTS = 900

# anomaly_detection.py
# number of actions (trajectories) used for decision
# NUM_ACTIONS = 10
