import airsim
import algorithm
import datetime
import numpy as np
import os
import pathlib
import PIL.Image as Image
import traceback
import util

#

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

#

# pole and roof
# START_POS = (0, 0, -5)
# TARGET_POS = (125, 25, -5)
# FLY_SPEED = 3
# DODGE_DIST = 3

# low fly
# START_POS = (-36, 1, -0.5)
# TARGET_POS = (-36, -50, -0.5)
# FLY_SPEED = 1.5
# DODGE_DIST = 1.5

# tree
START_POS = (134, 120, -5)
TARGET_POS = (120, 230, -5)
FLY_SPEED = 3
DODGE_DIST = 3

RECORD_PATH = r".\record"

#


def move_to_target(client):
    client.moveToPositionAsync(
        *TARGET_POS,
        velocity=FLY_SPEED,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False),
    )


def main():
    depth_model = tf.keras.models.load_model(
        filepath="./depth/model.h5",
        custom_objects={"relu6": tf.nn.relu6, "tf": tf.nn},
        compile=False,
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
    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(*START_POS), airsim.Quaternionr(0, 0, 0, 1),),
        ignore_collison=True,
    )
    move_to_target(client)
    #
    record = pathlib.Path(RECORD_PATH).joinpath(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    scene_record = record.joinpath("scene")
    depth_record = record.joinpath("depth")
    scene_record.mkdir(mode=0o755, parents=True)
    depth_record.mkdir(mode=0o755, parents=True)
    record_counter = 1
    #
    try:
        while True:
            pose = client.simGetVehiclePose().position
            if pose.distance_to(airsim.Vector3r(*TARGET_POS)) < 2:
                print("target position arrived")
                client.moveByVelocityAsync(0, 0, 0, 5).join()
                break
            #
            response = client.simGetImages(
                [
                    airsim.ImageRequest(
                        camera_name="front_center",
                        image_type=airsim.ImageType.Scene,
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
            result = algorithm.two_step_decision(depth)
            print(result)
            #
            filename = str(record_counter).zfill(3) + ".png"
            Image.frombytes(
                mode="RGB",
                size=(response[0].width, response[0].height),
                data=response[0].image_data_uint8,
            ).save(scene_record.joinpath(filename))
            util.save_algorithm(
                depth_data=depth,
                algo_result=result,
                path=depth_record.joinpath(filename),
            )
            record_counter += 1
            #
            dodge_duration = DODGE_DIST / FLY_SPEED
            if result == algorithm.Result.UP_RIGHT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4,
                    vy=FLY_SPEED / 2 ** 0.5,
                    vz=-FLY_SPEED / 2 ** 0.5,
                    duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.RIGHT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED, vy=FLY_SPEED, vz=0, duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.DOWN_RIGHT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4,
                    vy=FLY_SPEED / 2 ** 0.5,
                    vz=FLY_SPEED / 2 ** 0.5,
                    duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.DOWN:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4, vy=0, vz=FLY_SPEED, duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.DOWN_LEFT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4,
                    vy=-FLY_SPEED / 2 ** 0.5,
                    vz=FLY_SPEED / 2 ** 0.5,
                    duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.LEFT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED, vy=-FLY_SPEED, vz=0, duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.UP_LEFT:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4,
                    vy=-FLY_SPEED / 2 ** 0.5,
                    vz=-FLY_SPEED / 2 ** 0.5,
                    duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.UP:
                client.moveByVelocityAsync(
                    vx=FLY_SPEED / 4, vy=0, vz=-FLY_SPEED, duration=dodge_duration,
                ).join()
                move_to_target(client)
            elif result == algorithm.Result.STOP:
                pass
            else:
                move_to_target(client)
    except:
        traceback.print_exc()
    #
    client.reset()


if __name__ == "__main__":
    main()

