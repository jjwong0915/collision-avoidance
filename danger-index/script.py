import airsim
import dataloader
import dotenv
import matplotlib.pyplot as plt
import model
import numpy as np
import os
import pathlib
import PIL.Image as Image

dotenv.load_dotenv()

WIDTH, HEIGHT = int(os.getenv("WIDTH")), int(os.getenv("HEIGHT"))
DEPTH_MIN, DEPTH_MAX = int(os.getenv("DEPTH_MIN")), int(os.getenv("DEPTH_MAX"))


def visualize_depth(input_dir, output_dir):
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    #
    for depth_file in input_path.glob("**/*.pfm"):
        depth_data, _ = airsim.read_pfm(depth_file.absolute())
        #
        clipped_data = np.clip(depth_data, DEPTH_MIN, DEPTH_MAX)
        transformed_data = (
            DEPTH_MAX
            * (np.log2(clipped_data) - np.log2(DEPTH_MIN))
            / (np.log2(DEPTH_MAX) - np.log2(DEPTH_MIN))
        )
        normalized_data = transformed_data * (255 / DEPTH_MAX)
        inversed_data = 255 - normalized_data.astype(np.uint8)
        depth_image = Image.fromarray(inversed_data)
        resized_image = depth_image.resize((WIDTH, HEIGHT))
        #
        result_path = output_path.joinpath(
            depth_file.relative_to(input_path)
        ).with_suffix(".png")
        result_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        resized_image.save(result_path.absolute())


def predict_depth(weight_path, input_dir):
    input_path = pathlib.Path(input_dir)
    danger_dataloader = dataloader.DangerIndexDataloader((WIDTH, HEIGHT))
    danger_model = model.DangerIndexModel((WIDTH, HEIGHT, 1))
    danger_model.load_weights(weight_path)
    #
    predicted_index = []
    labeled_index = []
    for depth_image in input_path.glob("**/*.png"):
        depth_data = danger_dataloader.read_depth_image(depth_image.absolute())
        predicted_index.append(danger_model.predict(np.array([depth_data]))[0][0])
        labeled_index.append(int(depth_image.stem))
    #
    plt.hlines(0.5, 0, 50, "r")
    plt.plot(labeled_index, predicted_index, "bo")
    plt.axis([0, 50, 0, 1.2])
    plt.show()
