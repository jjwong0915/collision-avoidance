import airsim
import datetime
import os
import os.path
import tempfile
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# create temporary directory
image_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
print(f"temporary image folder: {image_dir}")

# fly to starting point
print("taking off...")
client.moveToPositionAsync(0, 0, -5, 5).join()
time.sleep(1)

# get camera image from the drone
print("starting to take image...")
for x in range(2, 22, 2):
    client.moveToPositionAsync(x, 0, -5, 1).join()
    time.sleep(1)
    image = client.simGetImage("front_center", airsim.ImageType.Scene)
    # save image to temporary directory
    filename = os.path.normpath(
        os.path.join(
            image_dir,
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + ".png",
        )
    )
    print(f"{filename} (size: {len(image)})")
    airsim.write_file(filename, image)

airsim.wait_key("Press any key to reset to original state")
client.reset()
