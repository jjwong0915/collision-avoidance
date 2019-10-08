import airsim
import collections
import datetime
import logging
import math
import os
import os.path
import random
import tempfile
import time

# configurable constants
DATASET_DIRECTORY = r'C:\Users\JJ Wong\Documents\NCTU\seminar\dataset'
COLLISION_NUMBER = 5
FRAME_PER_COLLISION = 60
FRAME_INTERVAL = 0.01
GENERATE_RANGE = ((-200, 200), (-200, 200))
FLYING_HEIGHT = 5
FLYING_VELOCITY = 5




def setRandomVehiclePose(self):
    # generate random position
    position = [
        random.uniform(r[0], r[1])
        for r in GENERATE_RANGE
    ] + [-FLYING_HEIGHT]
    orientation = random.uniform(-math.pi, math.pi)
    self.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(*position),
            airsim.Quaternionr(0, 0, math.sin(orientation/2), math.cos(orientation/2))
        ),
        ignore_collison = True,
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
        # start try to collide
        collision_index = 0
        while collision_index < COLLISION_NUMBER:
            direction = setRandomVehiclePose(client)
            while client.simGetCollisionInfo().has_collided:
                direction = setRandomVehiclePose(client)
            # move toward the direction
            client.moveByVelocityZAsync(
                vx = FLYING_VELOCITY*math.cos(direction),
                vy = FLYING_VELOCITY*math.sin(direction),
                z = -FLYING_HEIGHT,
                drivetrain = airsim.DrivetrainType.ForwardOnly,
                duration = 1000,
            )
            # initialize image queue
            image_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
            while True:
                # gather images from the camera
                image_queue.appendleft(
                    client.simGetImage('front_center', airsim.ImageType.Scene)
                )
                # check if a collision occured
                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided:
                    if len(image_queue) < FRAME_PER_COLLISION:
                        print(
                            f'not enough picture before this collision: '
                            f'{len(image_queue)}'
                        )
                        break
                    occured_time = datetime.datetime.now()
                    # save the capturede images just before collision occured
                    folder_name = str(occured_time.strftime('%Y-%m-%d-%H-%M-%S'))
                    folder_path = os.path.join(dataset_path, folder_name)
                    os.mkdir(folder_path)
                    for num in range(len(image_queue)):
                        image = image_queue.pop()
                        file_path = os.path.join(
                            folder_path,
                            str(num).zfill(3)+'.png'
                        )
                        airsim.write_file(file_path, image)
                    # finished this collision
                    print(f'collision data #{collision_index+1} is saved')
                    collision_index += 1
                    break
                # time.sleep(FRAME_INTERVAL)
        print(f'{COLLISION_NUMBER} collision datas are saved.')
        print('Returning to origin state...')
        time.sleep(5)
        client.reset()
    except:
        logging.exception('')
        client.reset()

main()
