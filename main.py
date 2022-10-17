import tensorflow as tf

if __name__ == "__main__":
  # create_sapimouse_actions.create_sapimouse_actions()
  # create_bezier_actions.create_bezier_actions()
  # create_equidistant_actions.create_equidistant_actions()

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  #
  # try:
  #   os.mkdir(stt.TRAINING_CURVES_PATH)
  #   os.mkdir(stt.TRAINED_MODELS_PATH)
  # except:
  #   print("Failed to create folders")
  #
  # # TRAIN and GENERATE ACTIONS
  # df_train = pd.read_csv("sapimouse_actions/actions_3min_dx_dy.csv", header=None)
  # for is_supervised in [True, False]:
  #   for loss_func_type in [LossFuncType.CUSTOM, LossFuncType.MSE]:
  #     for model_type in [ModelType.FCN, ModelType.BI_DIRECTIONAL]:
  #       model_name = stt.get_name(model_type=model_type, loss_func_type=loss_func_type,
  #                                 is_supervised=is_supervised)
  #
  #       encoder, autoencoder = autoencoder_training.train_autoencoder(df=df_train, input_size=128, input_dim=2,
  #                                                                     num_filters=128,
  #                                                                     batch_size=32,
  #                                                                     use_supervised=is_supervised,
  #                                                                     loss_func_type=loss_func_type,
  #                                                                     model_type=model_type, model_name=model_name,
  #                                                                     epochs=1)
  #
  #       generate_autoencoder_actions.generate_autoencoder_actions(input_size=128, input_dim=2, model_name=model_name)
  #
  # # EVALUATE
  # anomaly_detection_pyod.anomaly_detection()
