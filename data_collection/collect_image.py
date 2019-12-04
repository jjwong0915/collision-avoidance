import airsim
import collections
import datetime
import json
import logging
import math
import os
import os.path
import random
import tempfile
import time

# configurable constants
DATA_PATH = r"C:\Users\JJ Wong\Documents\NCTU\seminar\data"
COLLISION_COUNT = 5
COLLISION_FRAME = 50
FLYING_START = ((-200, 200), (-200, 200), (-3, -6))
FLYING_SPEED = 1
FLYING_DURATION = 100  # seconds


def main():
    # create data directory
    data_directory = os.path.join(
        DATA_PATH, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"),
    )
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    # make the multirotor take off
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    try:
        # initialize collision record
        collision_record = []
        while len(collision_record) < COLLISION_COUNT:
            # initialize flying position and orientation
            while True:
                position = [
                    random.uniform(start_range[0], start_range[1])
                    for start_range in FLYING_START
                ]
                orientation = random.uniform(-math.pi, math.pi)
                client.simSetVehiclePose(
                    airsim.Pose(
                        airsim.Vector3r(*position),
                        airsim.Quaternionr(
                            0, 0, math.sin(orientation / 2), math.cos(orientation / 2)
                        ),
                    ),
                    ignore_collison=True,
                )
                if not client.simGetCollisionInfo().has_collided:
                    break
            # initialize flying deadline
            deadline = datetime.datetime.now() + datetime.timedelta(
                seconds=FLYING_DURATION
            )
            # initialize image queue
            frame_queue = collections.deque(maxlen=COLLISION_FRAME)
            # start flying
            client.moveByVelocityZAsync(
                vx=FLYING_SPEED * math.cos(orientation),
                vy=FLYING_SPEED * math.sin(orientation),
                z=position[2],
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                duration=FLYING_DURATION,
            )
            while True:
                # gather images from the camera
                responses = client.simGetImages(
                    [
                        airsim.ImageRequest(
                            "front_center",
                            airsim.ImageType.Scene,
                            pixels_as_float=False,
                        ),
                        airsim.ImageRequest(
                            "front_center",
                            airsim.ImageType.DepthPlanner,
                            pixels_as_float=True,
                        ),
                        airsim.ImageRequest(
                            "front_center",
                            airsim.ImageType.DepthPerspective,
                            pixels_as_float=True,
                        ),
                        airsim.ImageRequest(
                            "front_center",
                            airsim.ImageType.Segmentation,
                            pixels_as_float=False,
                        ),
                    ]
                )
                # push images into queue
                frame_queue.appendleft(
                    {
                        "scene": responses[0].image_data_uint8,
                        "depth_planner": airsim.get_pfm_array(responses[1]),
                        "depth_perspective": airsim.get_pfm_array(responses[2]),
                        "segmentation": responses[3].image_data_uint8,
                    }
                )
                # check if a collision occured
                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    # check if the number of pictures is enough
                    if len(frame_queue) < COLLISION_FRAME:
                        print(
                            f"not enough picture before this collision: "
                            f"{len(frame_queue)}"
                        )
                        break
                    # save the collision info with custom timestamp
                    timestamp = str(
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    )
                    collision_record.append(
                        {
                            "normal": collision_info.normal.to_numpy_array().tolist(),
                            "impact_point": collision_info.impact_point.to_numpy_array().tolist(),
                            "position": collision_info.position.to_numpy_array().tolist(),
                            "penetration_depth": collision_info.penetration_depth,
                            "time_stamp": timestamp,
                            "object_name": collision_info.object_name,
                            "object_id": collision_info.object_id,
                        }
                    )
                    with open(
                        os.path.join(data_directory, "collision_record.json"), "w"
                    ) as record_file:
                        json.dump(collision_record, record_file)
                    # create image directories
                    scene_directory = os.path.join(data_directory, "scene", timestamp)
                    depth_planner_directory = os.path.join(
                        data_directory, "depth_planner", timestamp
                    )
                    depth_perspective_directory = os.path.join(
                        data_directory, "depth_perspective", timestamp
                    )
                    segmentation_directory = os.path.join(
                        data_directory, "segmentation", timestamp
                    )
                    os.makedirs(scene_directory)
                    os.makedirs(depth_planner_directory)
                    os.makedirs(depth_perspective_directory)
                    os.makedirs(segmentation_directory)
                    # save the captured images
                    frame_index = 1
                    while len(frame_queue) > 0:
                        frame = frame_queue.pop()
                        base_name = str(frame_index).zfill(3)
                        airsim.write_file(
                            os.path.join(scene_directory, base_name + ".png"),
                            frame["scene"],
                        )
                        airsim.write_pfm(
                            os.path.join(depth_planner_directory, base_name + ".pfm"),
                            frame["depth_planner"],
                        )
                        airsim.write_pfm(
                            os.path.join(depth_planner_directory, base_name + ".pfm"),
                            frame["depth_perspective"],
                        )
                        airsim.write_file(
                            os.path.join(segmentation_directory, base_name + ".png"),
                            frame["segmentation"],
                        )
                        frame_index += 1
                    # finish this collision
                    print(f"collision record #{len(collision_record)} is saved")
                    break
                elif datetime.datetime.now() > deadline:
                    print("overdue the deadline")
                    break
        print(f"{COLLISION_COUNT} collision datasets are saved.")
        print("returning to origin state...")
        time.sleep(5)
    except:
        logging.exception("")
    client.reset()


main()
