import tensorflow as tf
import pandas as pd
import os
from autoencoder_training import train_autoencoder
from generate_autoencoder_actions import generate_autoencoder_actions
from settings import create_directory, get_name, TRAINING_CURVES_FOLDER_NAME, TRAINED_MODELS_FOLDER_NAME, LossFuncType, \
    ModelType


def train():
    print(tf.__version__)

    work_dir = os.path.dirname(os.path.realpath(__file__))
    # work_dir = os.path.join()'/code'  # use os.getcwd() when running this locally instead of using docker
    # work_dir = os.getcwd()  # use os.getcwd() when running this locally instead of using docker
    data_dir = os.path.join(work_dir, "saved_data")

    training_models_path = os.path.join(work_dir, TRAINED_MODELS_FOLDER_NAME)
    training_curves_path = os.path.join(work_dir, TRAINING_CURVES_FOLDER_NAME)
    generated_actions_path = os.path.join(data_dir, 'generated_actions')
    equidistant_actions_path = os.path.join(data_dir, 'equidistant_actions')

    create_directory(training_models_path)
    create_directory(training_curves_path)
    create_directory(generated_actions_path)

    # TRAIN and GENERATE ACTIONS

    train_data_filename = "actions_3min_dx_dy.csv"
    supervised_feature_data_filename = "equidistant_3min.csv"

    df_train = pd.read_csv(os.path.join(data_dir, 'sapimouse_actions', train_data_filename), header=None)
    supervised_feature_df = pd.read_csv(os.path.join(data_dir, 'equidistant_actions', supervised_feature_data_filename),
                                        header=None)

    equidistant_actions_df = pd.read_csv(os.path.join(equidistant_actions_path, 'equidistant_1min.csv'), header=None)

    # for is_supervised in [True, False]:
    for is_supervised in [False]:
        # for loss_func_type in [LossFuncType.CUSTOM, LossFuncType.MSE]:
        for loss_func_type in [LossFuncType.CUSTOM]:
            # for model_type in [ModelType.FCN, ModelType.BI_DIRECTIONAL]:
            for model_type in [ModelType.BI_DIRECTIONAL]:
                model_name = get_name(model_type=model_type, loss_func_type=loss_func_type,
                                      is_supervised=is_supervised)

                # encoder, autoencoder = train_autoencoder(df=df_train,
                #                                          input_size=128, input_dim=2,
                #                                          num_filters=128,
                #                                          batch_size=32,
                #                                          use_supervised=is_supervised,
                #                                          loss_func_type=loss_func_type,
                #                                          model_type=model_type, model_name=model_name,
                #                                          epochs=1,
                #                                          training_models_path=training_models_path,
                #                                          supervised_feature_df=supervised_feature_df)

                generate_autoencoder_actions(input_size=128, input_dim=2,
                                             model_name=model_name,
                                             training_models_path=training_models_path,
                                             out_dir=generated_actions_path,
                                             equidistant_actions_df=equidistant_actions_df)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Current Dir: ", os.getcwd())
    train()
