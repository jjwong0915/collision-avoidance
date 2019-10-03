import airsim
import collections
import datetime
import math
import os
import os.path
import random
import tempfile
import time

# configurable constants
collision_number = 10
frame_per_collision = 30
frame_interval = 0.1
flying_velocity = 10
flying_height = 5
flying_direction_variation = math.pi/16


def moveByDirectionZ(self, v, dir, **kwargs):
    # TODO: turn camera to the front
    self.moveByVelocityZAsync(
        vx = v*math.sin(dir),
        vy = v*math.cos(dir),
        **kwargs
    ).join()


def capture_images(client):
    direction = 0
    collision_count = 0
    image_queue = collections.deque(maxlen=frame_per_collision)
    while collision_count < collision_number:
        # randomly change the direction
        direction += random.normalvariate(0, flying_direction_variation)
        # move toward the duration
        moveByDirectionZ(
            self = client,
            v = flying_velocity,
            dir = direction,
            z = -1*flying_height,
            duration = frame_interval,
        )
        # gather images from the camera
        image_queue.appendleft(
            client.simGetImage('front_center', airsim.ImageType.Scene)
        )
        # check if a collision occured
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided and len(image_queue) >= frame_per_collision:
            occured_time = datetime.datetime.now()
            print(
                f'collision dataset #{collision_count+1} captured at {occured_time}'
            )
            # save the capturede images just before collision occured
            folder_name = str(occured_time.strftime('%Y-%m-%d-%H-%M-%S'))
            folder_path = os.path.join(image_dir, folder_name)
            os.mkdir(folder_path)
            for num in range(len(image_queue)):
                image = image_queue.pop()
                file_path = os.path.join(folder_path, str(num)+'.png')
                airsim.write_file(file_path, image)
            # turn back and move a bit
            direction += math.pi + random.normalvariate(0, math.pi/4)
            moveByDirectionZ(
                self = client,
                v = flying_velocity,
                dir = direction,
                z = -1*flying_height,
                duration = frame_interval*5,
            )
            collision_count += 1

    print(f'{collision_number} collision datasets are saved.')

def main():
    # create temporary directory
    image_dir = os.path.join(
        tempfile.gettempdir(),
        'airsim_drone',
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
    )
    os.makedirs(image_dir)
    print (f'temporary image folder: {image_dir}')
    try:
        # connect to the AirSim simulator
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        capture_images(client)
        print('Returning to origin state...')
        time.sleep(5)
        client.reset()
    except Exception as e:
        print(e)
        client.reset()

main()
