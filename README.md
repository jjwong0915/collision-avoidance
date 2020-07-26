# Seminar in automatic drone collision avoidance system 
This repository consists of four directories, each of them represents a stage of the seminar.


## Common information

* All of the directories use [Pipenv](https://github.com/pypa/pipenv) to manage versions of Python packages. Using virtual environment for Python is recommended since each directory may uses different versions of same package.

* Files end with `.ipynb` are [Jupyter](https://jupyter.org/) notebooks. The `jupyter` package should be installed with `pipenv`, so you can use it directly.


## Data collection

* Introduction:
  
  This is the first stage of the seminar, which focuses on collecting photos from motirotor camera just before collisions. We use [Airsim](https://github.com/microsoft/AirSim) to simulate random collisions and generate ground-truth depth image.

* How to use this directory:
  
  1. Download and run a [compiled Airsim environment](https://github.com/microsoft/AirSim/releases) in background. We use "Neighborhood" as our main environment. You need to make sure your hardware is strong enough or the time interval between collected photos will get longer. The size of collected image can be configured with settings of Airsim.
  
  2. Install python pacakges and change the configurable constants in the head of `collect_image.py`.
      * `DATA_PATH`: The path where the script will create data folder.
      * `COLLISION_COUNT`: The number of collision data should be collected before the script ends.
      * `COLLISION_FRAME`: The number of frames in each collision data.
      * `FLYING_START`: The range of start point for the flying before each collision.
      * `FLYING_SPEED`: The flying speed before each collision.
      * `FLYING_DURATION`: The timeout for detecting the motirotor flyed over bound.
  
  3. Execute `collect_image.py` and check the Airsim environment in background should having actions. An directory with the current time as name should be created in `DATA_PATH`.


## Depth Prediction

* Introduction:

  Second stage is building a depth predicition model, which is mostly inherited from [the work](https://etd.lib.nctu.edu.tw/cgi-bin/gs32/tugsweb.cgi?o=dnctucdr&s=id=%22GT070656618%22.) of Mian-Jhong Chiu previously in our lab.

* How to use this directory:

  * `trainer/training_Airsim.ipynb`: This is the training script which generates `h5` files that contains trained parameters.

  * `main.ipynb`: This is the evaluation script which generates videos containing ground-truth depth images and predicted depth images for comparison.

  * `data`, `result`, `test`: These are dummy directory for putting image data. Images in these directories do not uploads to Github.

  * `depth_defer/get3Dpos.ipynb`: This is the experimenting script for the collision avoidance algorithm.


## Danger Index

* Introduction:

  Third stage is find a method to determine danger level of flying position. The danger-index method uses a nerual network to do the job, but it is then abandones due to the performance problem.

* How to use this directory:

  * `main.py`: This is the training script that generates a trained danger-index model.

  * `script.py`: The model contains two utility scripts. The `visualize_depth` function converts `pfm` depth images into gray-scaled `png` image and the `predict_danger` function evaluates the danger-index model by generating chart that compares ground-truth and prediction.


## Collision Avoidance

* Introduction:

  This final result of the seminar is the implementation of designed collision avoidance algorithm. The algorithm currently contain some parameters that needs to change in different situation.

* How to use this directory:

  1. Run an Airsim environment in the background.

  2. Set the parameters in the head of `algorithms.py`.
      * `IMAGE_WIDTH`, `IMAGE_HEIGHT`: The image width and height of input camera photos.
      * `FLIGHT_WIDTH`, `FLIGHT_HEIGHT`: The size of the flying motirotor.
      * `SAFE_DISTANCE`: How short of distance is allowed before start avoiding an obstable.
      * `DANGER_PIXEL_RATIO`: How tolerance is the algorithm for the error of depth predicting.
      * `DANGER_WEIGHT_SUM`: How tolerance is the algorithm for the amount of obstacle in a direction.

  3. Set the paramters in the head of `main.py`:
      * `TARGET_POS`: The target position of the flying.
      * `TARGET_SPEED`: The default flying speed of motirotor.

  3. Run `main.py` and you should see the motirotor start flying in the Airsim environment.


