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
DATASET_DIRECTORY = r'C:\Users\JJ Wong\Documents\NCTU\seminar\data'
COLLISION_NUMBER = 5
FRAME_PER_COLLISION = 60
GENERATE_RANGE = ((-200, 200), (-200, 200))
FLYING_RANGE = ((-400, 400), (-400, 400))
FLYING_HEIGHT = 5
FLYING_VELOCITY = 5


def setRandomVehiclePose(self):
    # generate random position
    position = [random.uniform(r[0], r[1])
                for r in GENERATE_RANGE] + [-FLYING_HEIGHT]
    orientation = random.uniform(-math.pi, math.pi)
    self.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(*position),
            airsim.Quaternionr(
                0,
                0,
                math.sin(orientation / 2),
                math.cos(orientation / 2),
            )),
        ignore_collison=True,
    )
    return orientation


def main():
    try:
        # create dataset directory
        dataset_path = os.path.join(
            DATASET_DIRECTORY,
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
        )
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        # connect to the AirSim simulator
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        # initialize to collide
        collision_data = {}
        collision_index = 0
        while collision_index < COLLISION_NUMBER:
            direction = setRandomVehiclePose(client)
            while client.simGetCollisionInfo().has_collided:
                direction = setRandomVehiclePose(client)
            # move toward the direction
            client.moveByVelocityZAsync(
                vx=FLYING_VELOCITY * math.cos(direction),
                vy=FLYING_VELOCITY * math.sin(direction),
                z=-FLYING_HEIGHT,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                duration=1000,
            )
            # initialize image queue
            scene_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
            segment_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
            depthvis_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
            while True:
                # gather images from the camera
                responses = client.simGetImages([
                    # png
                    airsim.ImageRequest(
                        'front_center',
                        airsim.ImageType.Scene,
                    ),
                    airsim.ImageRequest(
                        'front_center',
                        airsim.ImageType.Segmentation,
                    ),
                    # pfm
                    airsim.ImageRequest(
                        'front_center',
                        airsim.ImageType.DepthVis,
                        True,
                    ),
                ])
                # push images into queue
                scene_queue.appendleft(responses[0].image_data_uint8)
                segment_queue.appendleft(responses[1].image_data_uint8)
                depthvis_queue.appendleft(airsim.get_pfm_array(responses[2]))
                # TODO: check if flied too far
                # check if a collision occured
                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    # check if the number of pictures is enough
                    if len(scene_queue) < FRAME_PER_COLLISION:
                        print(f'not enough picture before this collision: '
                              f'{len(scene_queue)}')
                        break
                    # save the collision info
                    timestamp = str(
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                    collision_data[timestamp] = {
                        'normal':
                        collision_info.normal.to_numpy_array().tolist(),
                        'impact_point':
                        collision_info.impact_point.to_numpy_array().tolist(),
                        'position':
                        collision_info.position.to_numpy_array().tolist(),
                        'penetration_depth':
                        collision_info.penetration_depth,
                        'object_name':
                        collision_info.object_name,
                    }
                    # create image folders
                    scene_path = os.path.join(
                        dataset_path,
                        'scene',
                        timestamp,
                    )
                    segment_path = os.path.join(
                        dataset_path,
                        'segment',
                        timestamp,
                    )
                    depthvis_path = os.path.join(
                        dataset_path,
                        'depthvis',
                        timestamp,
                    )
                    os.makedirs(scene_path)
                    os.makedirs(segment_path)
                    os.makedirs(depthvis_path)
                    # save the captured images
                    for num in range(len(scene_queue)):
                        airsim.write_file(
                            os.path.join(
                                scene_path,
                                str(num).zfill(3) + '.png',
                            ), scene_queue.pop())
                        airsim.write_file(
                            os.path.join(
                                segment_path,
                                str(num).zfill(3) + '.png',
                            ), segment_queue.pop())
                        airsim.write_pfm(
                            os.path.join(
                                depthvis_path,
                                str(num).zfill(3) + '.pfm',
                            ), depthvis_queue.pop())
                    # finish this collision
                    print(f'collision data #{collision_index+1} is saved')
                    collision_index += 1
                    break
        # save collision data and exit
        with open(
                os.path.join(dataset_path, 'collision_data.json'),
                'w',
        ) as fd:
            json.dump(collision_data, fd)
        print(f'{COLLISION_NUMBER} collision datasets are saved.')
        print('Returning to origin state...')
        time.sleep(5)
        client.reset()
    except:
        logging.exception('')
        client.reset()


main()
