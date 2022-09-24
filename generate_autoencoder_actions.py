import pandas as pd
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from settings import TRAINED_MODELS_FOLDER_NAME

def generate_autoencoder_actions(input_size, input_dim, model_name: str, training_models_path, equidistant_actions_df, out_dir: str):

  # OUTPUT_DIR = 'generated_actions'
  # try:
  #   os.mkdir(OUTPUT_DIR)
  # except OSError:
  #   print('Directory %s already exists' % OUTPUT_DIR)
  # else:
  #   print('Successfuly created the directory %s' % OUTPUT_DIR)

  tic = time.perf_counter()
  # load model
  model_path = os.path.join(training_models_path, model_name)
  autoencoder = load_model(model_path, compile=False)
  # Generate mouse curves from fixed endpoints - equidistant sequence
  # df0 = pd.read_csv("equidistant_actions/equidistant_1min.csv", header=None)
  array = equidistant_actions_df.values
  nsamples, nfeatures = array.shape
  nfeatures = nfeatures - 1
  X = array[:, 0:nfeatures]
  X = X.reshape(-1, input_size, input_dim)
  # generate actions using the autoencoder
  df_generated = autoencoder.predict(X)
  dim1, dim2, dim3 = df_generated.shape
  df_generated = df_generated.reshape(dim1, dim2 * dim3)
  df_generated = pd.DataFrame(data=df_generated)
  df_generated = df_generated.apply(np.round)
  # Fix negative zeros issue
  df_generated[df_generated == 0.] = 0.
  filename = os.path.join(out_dir, model_name + ".csv")
  df_generated.to_csv(filename, index=False, header=False)
  toc = time.perf_counter()

  print(f"Execution time: {toc - tic:0.4f} seconds")


if __name__ == "__main__":
  generate_autoencoder_actions()
