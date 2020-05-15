import tensorflow as tf

depth_model = tf.keras.models.load_model(
    filepath="./depth/model.h5", custom_objects={"relu6": tf.nn.relu6, "tf": tf.nn},
)
depth_model.summary()
