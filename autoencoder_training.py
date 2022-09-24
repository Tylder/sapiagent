import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from settings import LossFuncType, ModelType
from sklearn.model_selection import train_test_split
from plots import plot_history
from autoencoder_models import bidirectional_autoencoder, fcn_autoencoder
import settings as stt


def train_autoencoder(df, input_size, input_dim, num_filters, batch_size, use_supervised, loss_func_type: LossFuncType,
                      model_type: ModelType, supervised_feature_df, model_name: str, training_models_path, epochs=100,
                      supervised_data_file_path: str = os.path.join(os.getcwd(), "saved_data", "equidistant_actions",
                                                                    "equidistant_3min.csv")):
    # split dataframe
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]

    # equidistant segments
    if use_supervised:
        # df_synth = pd.read_csv("equidistant_actions/equidistant_3min.csv", header=None)
        # df_synth = pd.read_csv(supervised_data_file_path, header=None)
        df_synth = supervised_feature_df
        X_synth = df_synth.values[:, :-1]

    X = X.reshape(-1, input_size, input_dim)
    if use_supervised:
        X_synth = X_synth.reshape(-1, input_size, input_dim)
        X_train, X_val, y_train, y_val, X_train_synth, X_val_synth, = train_test_split(X, y, X_synth, test_size=0.25,
                                                                                       random_state=stt.RANDOM_STATE)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=stt.RANDOM_STATE)

    mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

    # convert to tensorflow dataset
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    if use_supervised:
        X_synth = np.asarray(X_synth).astype(np.float32)

    if use_supervised:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_synth, X_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_synth, X_val))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    start_time = time.time()
    if model_type == ModelType.FCN:
        encoder, model = fcn_autoencoder(input_shape=(input_size, input_dim), loss_func_type=loss_func_type,
                                         fcn_filters=num_filters, features=128, dimensions=2, bottleneck=True)
    if model_type == ModelType.BI_DIRECTIONAL:
        encoder, model = bidirectional_autoencoder(input_size=input_size, input_dim=input_dim, loss_func_type=loss_func_type)
    # train.py model
    history = model.fit(train_ds, epochs=epochs, shuffle=False, validation_data=val_ds)
    duration = time.time() - start_time
    print("Training duration: " + str(duration / 60))
    print(model_name)
    plot_history(history, model_name)
    model_path = os.path.join(training_models_path, model_name)
    model.save(model_path)
    # model.save(stt.TRAINED_MODELS_FOLDER_NAME + "/" + model_name)
    return encoder, model


if __name__ == "__main__":
    try:
        os.mkdir(stt.TRAINING_CURVES_FOLDER_NAME)
        os.mkdir(stt.TRAINED_MODELS_FOLDER_NAME)
    except OSError:
        print('Model will be saved in folder ' + TRAINED_MODELS_PATH)
        print('Training curve plot will be saved in folder ' + TRAINING_CURVES_PATH)
    if stt.TRAINING:
        # training data
        data_dir = os.path.join(os.getcwd(), "saved_data")

        train_data_filename = "actions_3min_dx_dy.csv"
        supervised_feature_data_filename = "equidistant_3min.csv"

        df_train = pd.read_csv(os.path.join(data_dir, 'sapimouse_actions', train_data_filename), header=None)
        supervised_feature_df = pd.read_csv(
            os.path.join(data_dir, 'equidistant_actions', supervised_feature_data_filename),
            header=None)

        encoder, autoencoder = train_autoencoder(df=df_train,
                                                 input_size=128, input_dim=2,
                                                 num_filters=128,
                                                 batch_size=32,
                                                 use_supervised=is_supervised,
                                                 loss_func_type=loss_func_type,
                                                 model_type=model_type, model_name=model_name,
                                                 epochs=1,
                                                 supervised_feature_df=supervised_feature_df)
    else:
        print('Set TRAINING to True in settings.py!')
