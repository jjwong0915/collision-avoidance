import os
import airsim
import pathlib
import numpy as np
import PIL.Image as Image

DEPTH_MAX = int(os.getenv("DEPTH_MAX"))
DEPTH_MIN = int(os.getenv("DEPTH_MIN"))
WIDTH, HEIGHT = int(os.getenv("INPUT_WIDTH")), int(os.getenv("INPUT_HEIGHT"))


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
