import tensorflow as tf
import airsim
import traceback
import algorithm
import numpy as np
import PIL.Image as Image

TARGET_POS = (40, 28, -10)
TARGET_SPEED = 2


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
    client.takeoffAsync().join()
    print("multirotor took off")
    #
    client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
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
            result = algorithm.two_step_decision(depth)
            print(result)
            if result == algorithm.Result.UP_RIGHT:
                client.moveByVelocityAsync(0, 1, -1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.RIGHT:
                client.moveByVelocityAsync(0, 1, 0, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.DOWN_RIGHT:
                client.moveByVelocityAsync(0, 1, 1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.DOWN:
                client.moveByVelocityAsync(0, 0, 1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.DOWN_LEFT:
                client.moveByVelocityAsync(0, -1, 1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.LEFT:
                client.moveByVelocityAsync(0, -1, 0, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.UP_LEFT:
                client.moveByVelocityAsync(0, -1, -1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.UP:
                client.moveByVelocityAsync(0, 0, -1, 1).join()
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)
            elif result == algorithm.Result.STOP:
                client.moveByVelocityAsync(0, 0, 0, 1).join()
            else:
                client.moveToPositionAsync(*TARGET_POS, TARGET_SPEED)

    except:
        traceback.print_exc()
    #
    client.reset()


if __name__ == "__main__":
    main()

