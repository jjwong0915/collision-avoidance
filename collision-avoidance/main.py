import tensorflow as tf
import airsim
import traceback
import algorithm
import numpy as np
import PIL.Image as Image

TARGET_POS = (25, 5, -5)


def main():
    depth_model = tf.keras.models.load_model(
        filepath="./depth/model.h5", custom_objects={"relu6": tf.nn.relu6, "tf": tf.nn},
    )
    print("depth model loaded")
    #
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("multirotor took off")
    #
    client.moveToPositionAsync(*TARGET_POS, 1)
    #
    try:
        while True:
            response = client.simGetImages(
                [
                    airsim.ImageRequest(
                        camera_name="front_center",
                        image_type=airsim.ImageType.Segmentation,
                        compress=False,
                    )
                ]
            )
            scene = airsim.string_to_uint8_array(response[0].image_data_uint8).reshape(
                response[0].height, response[0].width, 3
            )
            depth = np.maximum(
                np.squeeze(depth_model.predict(np.expand_dims(scene, axis=0))[0]), 1
            )
            #
            # visualized_depth = np.uint8(np.minimum(depth, 128) * 2)
            # Image.fromarray(visualized_depth, "L").show()
            #
            result = algorithm.projection_approach(depth)
            print(result)
    except:
        traceback.print_exc()
    #
    client.reset()


if __name__ == "__main__":
    main()

