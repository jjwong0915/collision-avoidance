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
DATASET_DIRECTORY = 'D:\\0612255\\108-1\\Project\\pics'
COLLISION_NUMBER = 5
FRAME_PER_COLLISION = 60
FRAME_INTERVAL = 0.01
GENERATE_RANGE = ((-200, 200), (-200, 200))
FLYING_HEIGHT = 5
FLYING_VELOCITY = 5
image_path = (DATASET_DIRECTORY+'\\image')
depthplanner_path = DATASET_DIRECTORY+'\\depth_planner'
depthvis_path = DATASET_DIRECTORY+'\\depth_vis'
segmentation_path = DATASET_DIRECTORY+'\\segmentation'
		

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
			depthplanner_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
			depthvis_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
			segmentation_queue = collections.deque(maxlen=FRAME_PER_COLLISION)
			while True:
				# gather images from the camera
				responses = client.simGetImages([
						airsim.ImageRequest(0, airsim.ImageType.DepthVis, True), #pfm
						airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, True),
						airsim.ImageRequest(0, airsim.ImageType.Segmentation), #png
						airsim.ImageRequest(0, airsim.ImageType.Scene)])
				
				for i, response in enumerate(responses):
					if response.pixels_as_float: #pfm format
						if i == 0:
							depthvis_queue.appendleft(airsim.get_pfm_array(response))
						elif i == 1:
							depthplanner_queue.appendleft(airsim.get_pfm_array(response))
					elif response.compress: #png format
						if i == 2:
							segmentation_queue.appendleft(response.image_data_uint8)
						elif i == 3:
							image_queue.appendleft(response.image_data_uint8)
				
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
					os.mkdir(os.path.join(image_path, folder_name))
					os.mkdir(os.path.join(depthplanner_path, folder_name))
					os.mkdir(os.path.join(depthvis_path, folder_name))
					os.mkdir(os.path.join(segmentation_path, folder_name))

					for num in range(len(image_queue)):
						airsim.write_file(
							os.path.join(image_path, folder_name, str(num).zfill(3)+'.png')
							, image_queue.pop() )
						airsim.write_file(
							os.path.join(segmentation_path, folder_name, str(num).zfill(3)+'.png')
							, segmentation_queue.pop() )
						airsim.write_pfm(
							os.path.join(depthplanner_path, folder_name, str(num).zfill(3)+'.pfm')
							, depthplanner_queue.pop() )
						airsim.write_pfm(
							os.path.join(depthvis_path, folder_name, str(num).zfill(3)+'.pfm')
							, depthvis_queue.pop() )
					
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